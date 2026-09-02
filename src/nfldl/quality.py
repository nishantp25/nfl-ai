"""Post-ingest validation.

Ingesting data that is wrong is worse than not ingesting it, because the failure
surfaces months later as an inexplicable model. These checks run against what is
actually on disk and produce a report that is meant to be read, not just stored.

Checks:
  PK uniqueness            the declared primary key really is one
  Null rates               per-column fill, flagged when a column is nearly empty
  Row-count sanity         season-over-season swings that suggest a partial file
  Season contiguity        holes inside the verified range
  Referential integrity    game ids in child tables exist in the canonical games
  Coverage honesty         a required dataset that silently lacks target seasons
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import polars as pl

from .config import QUALITY_REPORT_PATH, RAW_DIR, get_settings
from .manifest import DatasetSpec, get_manifest
from .storage import INGEST_COLUMNS, read_dataset

log = logging.getLogger(__name__)

# Row-count change between adjacent seasons beyond this is suspicious. The real
# structural changes (2020's no-preseason, the 17-game move in 2021, playoff
# expansion) are all well inside 35%.
ROW_COUNT_TOLERANCE = 0.35

# A column this empty is almost certainly "did not exist that season" rather
# than "occasionally missing", and must not be imputed.
NEAR_EMPTY_THRESHOLD = 0.98


@dataclass
class Check:
    name: str
    dataset: str
    passed: bool
    severity: str  # "error" | "warning" | "info"
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


def _feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in INGEST_COLUMNS]


def _seasons_present(df: pl.DataFrame) -> list[int]:
    """Season values as ints. nflverse ships this column as float in a few
    tables, and a float season silently breaks every range comparison."""
    return sorted(
        int(s) for s in df.get_column("season").drop_nulls().unique().to_list()
    )


def check_primary_key(df: pl.DataFrame, spec: DatasetSpec) -> Check:
    pk = list(spec.primary_key)
    absent = [c for c in pk if c not in df.columns]
    if absent:
        return Check(
            "primary_key", spec.name, False, "error",
            f"declared key columns not in the data: {absent}",
            {"declared": pk, "absent": absent},
        )
    if not pk:
        return Check("primary_key", spec.name, True, "info", "no key declared")

    n_unique = df.select(pk).unique().height
    dupes = df.height - n_unique
    if dupes == 0:
        return Check("primary_key", spec.name, True, "info", f"{pk} unique across {df.height:,} rows")

    example = (
        df.group_by(pk).agg(pl.len().alias("n")).filter(pl.col("n") > 1).sort("n", descending=True).head(3)
    )
    return Check(
        "primary_key", spec.name, False, "error",
        f"{dupes:,} duplicate rows under {pk}",
        {"duplicates": dupes, "worst": example.to_dicts()},
    )


def check_null_rates(df: pl.DataFrame, spec: DatasetSpec) -> Check:
    cols = _feature_columns(df)
    if not cols or df.height == 0:
        return Check("null_rates", spec.name, True, "info", "nothing to measure")

    nulls = df.select(cols).null_count().row(0)
    rates = {c: n / df.height for c, n in zip(cols, nulls)}
    near_empty = {c: round(r, 4) for c, r in rates.items() if r >= NEAR_EMPTY_THRESHOLD}

    # Key columns being null is always an error; a sparse optional column is not.
    key_nulls = {c: round(rates[c], 4) for c in spec.primary_key if rates.get(c, 0) > 0}
    if key_nulls:
        return Check(
            "null_rates", spec.name, False, "error",
            f"primary key columns contain nulls: {key_nulls}",
            {"key_nulls": key_nulls, "near_empty": near_empty},
        )

    if near_empty:
        return Check(
            "null_rates", spec.name, True, "warning",
            f"{len(near_empty)} columns are >={int(NEAR_EMPTY_THRESHOLD*100)}% null -- treat as "
            f"'not collected', do not impute: {sorted(near_empty)[:8]}",
            {"near_empty": near_empty},
        )
    return Check("null_rates", spec.name, True, "info", f"{len(cols)} columns, no near-empty ones")


def check_row_counts(df: pl.DataFrame, spec: DatasetSpec) -> Check:
    if "season" not in df.columns:
        return Check("row_counts", spec.name, True, "info", "not season-partitioned")

    counts = (
        df.group_by("season").agg(pl.len().alias("rows")).sort("season")
    )
    rows = counts.to_dicts()
    anomalies = []
    for prev, cur in zip(rows, rows[1:]):
        if prev["rows"] == 0:
            continue
        delta = abs(cur["rows"] - prev["rows"]) / prev["rows"]
        if delta > ROW_COUNT_TOLERANCE:
            anomalies.append(
                {"from": prev["season"], "to": cur["season"],
                 "from_rows": prev["rows"], "to_rows": cur["rows"], "change": round(delta, 3)}
            )
    if anomalies:
        return Check(
            "row_counts", spec.name, True, "warning",
            f"{len(anomalies)} season-over-season row-count jumps >{int(ROW_COUNT_TOLERANCE*100)}%; "
            "check for a partial download or a genuine format change",
            {"anomalies": anomalies},
        )
    return Check("row_counts", spec.name, True, "info", f"{len(rows)} seasons, stable volumes")


def check_season_contiguity(df: pl.DataFrame, spec: DatasetSpec) -> Check:
    if "season" not in df.columns or df.height == 0:
        return Check("contiguity", spec.name, True, "info", "not applicable")
    present = _seasons_present(df)
    holes = sorted(set(range(present[0], present[-1] + 1)) - set(present))
    if holes:
        return Check(
            "contiguity", spec.name, False, "error",
            f"missing seasons inside the ingested range: {holes}",
            {"present": present, "holes": holes},
        )
    return Check("contiguity", spec.name, True, "info", f"contiguous {present[0]}-{present[-1]}")


def check_target_coverage(df: pl.DataFrame, spec: DatasetSpec) -> Check:
    cfg = get_settings()
    if "season" not in df.columns:
        return Check("target_coverage", spec.name, True, "info", "not season-partitioned")
    present = set(_seasons_present(df))
    gap = sorted(set(cfg.target_seasons) - present)
    if not gap:
        return Check("target_coverage", spec.name, True, "info", "full target window present")
    severity = "warning" if not spec.required else "warning"
    return Check(
        "target_coverage", spec.name, True, severity,
        f"{len(gap)} target seasons absent: {gap}. This is coverage, not corruption -- features "
        "from this dataset must be null there.",
        {"missing_seasons": gap},
    )


def check_referential_integrity(df: pl.DataFrame, spec: DatasetSpec) -> Check:
    """Child tables must reference games that exist in the canonical game table."""
    game_cols = [c for c in ("game_id", "nflverse_game_id") if c in df.columns]
    if not game_cols or spec.name == "schedules":
        return Check("referential_integrity", spec.name, True, "info", "no game reference")

    col = game_cols[0]
    try:
        games = read_dataset(get_manifest()["schedules"], root=RAW_DIR)
    except FileNotFoundError:
        return Check(
            "referential_integrity", spec.name, True, "warning",
            "schedules not ingested; cannot check game references",
        )

    known = set(games.get_column("game_id").to_list())
    referenced = set(df.get_column(col).drop_nulls().to_list())
    orphans = sorted(referenced - known)
    if orphans:
        return Check(
            "referential_integrity", spec.name, False, "error",
            f"{len(orphans)} game ids not present in schedules (e.g. {orphans[:3]})",
            {"orphan_count": len(orphans), "examples": orphans[:10]},
        )
    return Check(
        "referential_integrity", spec.name, True, "info",
        f"all {len(referenced):,} referenced games exist in schedules",
    )


CHECKS = (
    check_primary_key,
    check_null_rates,
    check_row_counts,
    check_season_contiguity,
    check_target_coverage,
    check_referential_integrity,
)


def validate_dataset(name: str) -> list[Check]:
    spec = get_manifest()[name]
    try:
        df = read_dataset(spec, root=RAW_DIR)
    except FileNotFoundError as exc:
        severity = "error" if spec.required else "info"
        return [Check("present", name, not spec.required, severity, str(exc))]

    results = [Check("present", name, True, "info", f"{df.height:,} rows, {df.width} columns")]
    for check in CHECKS:
        try:
            results.append(check(df, spec))
        except Exception as exc:  # a broken check must not mask the others
            results.append(Check(check.__name__, name, False, "error", f"check crashed: {exc}"))
    return results


def validate_all(names: list[str] | None = None) -> dict[str, Any]:
    mf = get_manifest()
    names = names or mf.names()
    all_checks: list[Check] = []
    for name in names:
        log.info("validating %s", name)
        all_checks.extend(validate_dataset(name))

    errors = [c for c in all_checks if not c.passed and c.severity == "error"]
    warnings = [c for c in all_checks if c.severity == "warning"]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets_checked": names,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "checks": [asdict(c) for c in all_checks],
    }
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"quality report {report['generated_at'][:19]}Z -- "
        f"{report['n_errors']} error(s), {report['n_warnings']} warning(s)",
        "-" * 78,
    ]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for c in report["checks"]:
        by_dataset.setdefault(c["dataset"], []).append(c)

    for dataset, checks in sorted(by_dataset.items()):
        worst = "ERROR" if any(not c["passed"] and c["severity"] == "error" for c in checks) else (
            "WARN" if any(c["severity"] == "warning" for c in checks) else "ok"
        )
        lines.append(f"[{worst:>5}] {dataset}")
        for c in checks:
            if c["severity"] == "info" and c["passed"] and c["name"] != "present":
                continue
            lines.append(f"         {c['name']}: {c['detail']}")
    return "\n".join(lines)
