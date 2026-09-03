"""Ingestion orchestration.

Rules this enforces at run time, not just in documentation:

* A season outside **verified** coverage is skipped with a reason, never
  fabricated and never silently absent from the report.
* Dependencies are ingested first (schedules -> stadiums -> weather).
* Failures are per-partition. One bad season does not abort a 15-season run,
  and the failure is recorded rather than swallowed.
* Re-running is cheap: existing partitions are skipped unless ``--force``.
* Partition identity is preserved in the rows themselves. If an upstream
  schema omits a partition column such as ``season``, ingestion restores it
  from the requested partition. If upstream supplies a conflicting value,
  ingestion fails rather than silently rewriting history.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import polars as pl

from .config import RAW_DIR, get_settings, nfl_season_of
from .manifest import DatasetSpec, get_manifest
from .registry import get_adapter, resolve_order
from .storage import partition_exists, write_partition

log = logging.getLogger(__name__)


@dataclass
class PartitionResult:
    dataset: str
    partition: dict[str, Any]
    status: str  # ingested | skipped_exists | skipped_unavailable | empty | failed
    rows: int = 0
    seconds: float = 0.0
    reason: str = ""


@dataclass
class IngestReport:
    started_at: str
    finished_at: str = ""
    results: list[PartitionResult] = field(
        default_factory=list
    )

    @property
    def failures(self) -> list[PartitionResult]:
        return [
            result
            for result in self.results
            if result.status == "failed"
        ]

    @property
    def ingested(self) -> list[PartitionResult]:
        return [
            result
            for result in self.results
            if result.status == "ingested"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "n_ingested": len(self.ingested),
            "n_failed": len(self.failures),
            "total_rows": sum(
                result.rows
                for result in self.results
            ),
            "results": [
                asdict(result)
                for result in self.results
            ],
        }


def attach_partition_keys(
    df: pl.DataFrame,
    partition_keys: dict[str, Any],
) -> pl.DataFrame:
    """Ensure partition identity is represented inside the data.

    Some upstream nflverse schemas omit a column that still forms part of our
    storage partition. The 2025+ depth-chart feed, for example, is requested as
    a season-specific asset but no longer contains a ``season`` column.

    A missing partition column is therefore restored from the partition request.

    An existing partition column is never silently overwritten. Every row must
    already agree with the requested partition value. A disagreement, including
    a null partition value, is treated as a source/partition integrity failure.

    This makes a stored Parquet file self-describing instead of relying on its
    directory name to recover essential identity later.
    """
    if not partition_keys:
        return df

    out = df

    for column, expected in partition_keys.items():
        if column not in out.columns:
            out = out.with_columns(
                pl.lit(expected).alias(column)
            )
            continue

        values = (
            out.get_column(column)
            .unique()
            .to_list()
        )

        conflicting = [
            value
            for value in values
            if (
                value is None
                or value != expected
            )
        ]

        if conflicting:
            raise ValueError(
                f"partition column '{column}' conflicts with requested "
                f"partition value {expected!r}; provider supplied "
                f"{conflicting!r}"
            )

    return out


def _seasons_for(
    spec: DatasetSpec,
    requested: list[int] | None,
) -> tuple[list[int], list[int]]:
    """Split requested seasons into available/unavailable verified coverage."""
    cfg = get_settings()

    wanted = (
        requested
        if requested is not None
        else cfg.target_seasons
    )

    current = nfl_season_of()

    coverage = (
        spec.verified
        if spec.verified.start is not None
        else spec.declared
    )

    available = coverage.clip(
        wanted,
        current_season=current,
    )

    unavailable = sorted(
        set(wanted)
        - set(available)
    )

    return (
        available,
        unavailable,
    )


def _partitions_for(
    spec: DatasetSpec,
    seasons: list[int],
) -> list[dict[str, Any]]:
    """Enumerate the partition key-sets this dataset needs."""
    variants = (
        list(spec.variants)
        or [None]
    )

    if not spec.is_partitioned_by_season:
        if "{stat_type}" in spec.local_storage_path:
            return [
                {
                    "stat_type": variant,
                }
                for variant in variants
                if variant
            ]

        return [
            {}
        ]

    parts: list[dict[str, Any]] = []

    for season in seasons:
        if "{stat_type}" in spec.local_storage_path:
            parts.extend(
                {
                    "season": season,
                    "stat_type": variant,
                }
                for variant in variants
                if variant
            )
        else:
            parts.append(
                {
                    "season": season,
                }
            )

    return parts


def ingest_dataset(
    name: str,
    *,
    seasons: list[int] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[PartitionResult]:
    mf = get_manifest()
    cfg = get_settings()

    spec = mf[name]
    adapter = get_adapter(name)

    results: list[PartitionResult] = []

    available, unavailable = _seasons_for(
        spec,
        seasons,
    )

    if unavailable:
        msg = (
            f"outside verified coverage ({spec.verified}); "
            f"{spec.coverage_caveat.strip()[:160] or 'no data exists here'}"
        )

        for season in unavailable:
            results.append(
                PartitionResult(
                    name,
                    {
                        "season": season,
                    },
                    "skipped_unavailable",
                    reason=msg,
                )
            )

        log.warning(
            "%s: %d requested season(s) unavailable and skipped: %s",
            name,
            len(unavailable),
            unavailable,
        )

        if (
            cfg.strict_coverage
            and spec.required
            and not available
        ):
            raise RuntimeError(
                f"required dataset '{name}' has no verified coverage for "
                f"the requested seasons {seasons or cfg.target_seasons}. "
                "Run `nfldl verify` before ingesting."
            )

    for keys in _partitions_for(
        spec,
        available,
    ):
        if (
            not force
            and partition_exists(
                spec,
                root=RAW_DIR,
                **keys,
            )
        ):
            results.append(
                PartitionResult(
                    name,
                    keys,
                    "skipped_exists",
                )
            )
            continue

        if dry_run:
            results.append(
                PartitionResult(
                    name,
                    keys,
                    "ingested",
                    reason="dry run",
                )
            )
            continue

        started = time.perf_counter()

        try:
            df = adapter.fetch(
                **keys
            )
        except Exception as exc:  # noqa: BLE001 - isolate provider partition failures
            log.error(
                "%s %s failed: %s",
                name,
                keys,
                exc,
            )

            results.append(
                PartitionResult(
                    name,
                    keys,
                    "failed",
                    seconds=(
                        time.perf_counter()
                        - started
                    ),
                    reason=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )
            continue

        elapsed = (
            time.perf_counter()
            - started
        )

        if df.is_empty():
            results.append(
                PartitionResult(
                    name,
                    keys,
                    "empty",
                    seconds=elapsed,
                    reason="provider returned zero rows",
                )
            )

            log.warning(
                "%s %s returned no rows",
                name,
                keys,
            )
            continue

        try:
            df = attach_partition_keys(
                df,
                keys,
            )
        except ValueError as exc:
            log.error(
                "%s %s partition metadata conflict: %s",
                name,
                keys,
                exc,
            )

            results.append(
                PartitionResult(
                    name,
                    keys,
                    "failed",
                    seconds=elapsed,
                    reason=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )
            continue

        record = write_partition(
            df,
            spec,
            root=RAW_DIR,
            notes=spec.coverage_caveat.strip()[:300],
            **keys,
        )

        results.append(
            PartitionResult(
                name,
                keys,
                "ingested",
                rows=record.rows,
                seconds=elapsed,
            )
        )

    return results


def ingest(
    names: list[str] | None = None,
    *,
    seasons: list[int] | None = None,
    force: bool = False,
    dry_run: bool = False,
    required_only: bool = False,
) -> IngestReport:
    mf = get_manifest()

    if names:
        targets = names
    elif required_only:
        targets = mf.required_names()
    else:
        targets = mf.names()

    ordered = resolve_order(
        targets
    )

    if ordered != targets:
        log.info(
            "dependency-ordered plan: %s",
            " -> ".join(ordered),
        )

    report = IngestReport(
        started_at=datetime.now(
            timezone.utc
        ).isoformat()
    )

    for name in ordered:
        log.info(
            "=== %s ===",
            name,
        )

        report.results.extend(
            ingest_dataset(
                name,
                seasons=seasons,
                force=force,
                dry_run=dry_run,
            )
        )

    report.finished_at = datetime.now(
        timezone.utc
    ).isoformat()

    return report


def format_report(
    report: IngestReport,
) -> str:
    by_dataset: dict[str, dict[str, int]] = {}
    rows: dict[str, int] = {}

    for result in report.results:
        bucket = by_dataset.setdefault(
            result.dataset,
            {},
        )

        bucket[result.status] = (
            bucket.get(
                result.status,
                0,
            )
            + 1
        )

        rows[result.dataset] = (
            rows.get(
                result.dataset,
                0,
            )
            + result.rows
        )

    lines = [
        (
            f"ingest {report.started_at[:19]}Z "
            f"-> {report.finished_at[:19]}Z  "
            f"({len(report.ingested)} partitions, "
            f"{sum(result.rows for result in report.results):,} rows)"
        ),
        "-" * 78,
    ]

    for dataset, counts in sorted(
        by_dataset.items()
    ):
        summary = ", ".join(
            f"{key}={value}"
            for key, value in sorted(
                counts.items()
            )
        )

        lines.append(
            f"  {dataset:18s} "
            f"{summary}  "
            f"rows={rows[dataset]:,}"
        )

    if report.failures:
        lines.append("")
        lines.append(
            "FAILURES:"
        )

        for failure in report.failures:
            lines.append(
                f"  {failure.dataset} "
                f"{failure.partition}: "
                f"{failure.reason}"
            )

    skipped = [
        result
        for result in report.results
        if result.status == "skipped_unavailable"
    ]

    if skipped:
        lines.append("")
        lines.append(
            f"{len(skipped)} partition(s) skipped as unavailable "
            "-- this is coverage, not error."
        )

    return "\n".join(
        lines
    )