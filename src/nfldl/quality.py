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

Datasets with known schema epochs are validated season by season. This prevents
columns that legitimately disappeared after an upstream schema change from being
treated as broken key fields in later seasons.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import polars as pl

from .config import QUALITY_REPORT_PATH, RAW_DIR, get_settings
from .manifest import DatasetSpec, ManifestError, get_manifest
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
    data: dict[str, Any] = field(
        default_factory=dict
    )


def _feature_columns(
    df: pl.DataFrame,
) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in INGEST_COLUMNS
    ]


def _seasons_present(
    df: pl.DataFrame,
) -> list[int]:
    """Return season values as ints.

    nflverse ships this column as a float in a few tables, and a float season
    silently breaks range comparisons.
    """
    return sorted(
        int(season)
        for season in (
            df.get_column("season")
            .drop_nulls()
            .unique()
            .to_list()
        )
    )


def _season_frame(
    df: pl.DataFrame,
    season: int,
) -> pl.DataFrame:
    """Return rows belonging to one NFL season.

    Casting allows sources that represent season as Int32, Int64, or a whole
    number float to be handled consistently.
    """
    return df.filter(
        pl.col("season")
        .cast(pl.Int64, strict=False)
        .eq(season)
    )


def _schema_seasons(
    df: pl.DataFrame,
    spec: DatasetSpec,
) -> tuple[list[int], Check | None]:
    """Resolve seasons needed for schema-aware validation.

    Returns ``(seasons, error)``. Datasets without schema epochs return an empty
    season list because callers should use the historical global contract.
    """
    if not spec.schema_epochs:
        return [], None

    if "season" not in df.columns:
        return [], Check(
            "schema_contract",
            spec.name,
            False,
            "error",
            "dataset declares schema epochs but the data has no season column",
        )

    season_column = df.get_column("season")

    if season_column.null_count() > 0:
        return [], Check(
            "schema_contract",
            spec.name,
            False,
            "error",
            (
                "dataset declares schema epochs but "
                f"{season_column.null_count():,} row(s) have null season"
            ),
            {
                "null_season_rows": season_column.null_count(),
            },
        )

    seasons = _seasons_present(df)

    for season in seasons:
        try:
            spec.schema_for_season(
                season
            )
        except ManifestError as exc:
            return [], Check(
                "schema_contract",
                spec.name,
                False,
                "error",
                str(exc),
                {
                    "season": season,
                },
            )

    return seasons, None


def _global_primary_key_check(
    df: pl.DataFrame,
    spec: DatasetSpec,
) -> Check:
    """Historical single-contract primary-key validation."""
    pk = list(
        spec.primary_key
    )

    absent = [
        column
        for column in pk
        if column not in df.columns
    ]

    if absent:
        return Check(
            "primary_key",
            spec.name,
            False,
            "error",
            f"declared key columns not in the data: {absent}",
            {
                "declared": pk,
                "absent": absent,
            },
        )

    if not pk:
        return Check(
            "primary_key",
            spec.name,
            True,
            "info",
            "no key declared",
        )

    n_unique = (
        df.select(pk)
        .unique()
        .height
    )

    dupes = (
        df.height
        - n_unique
    )

    if dupes == 0:
        return Check(
            "primary_key",
            spec.name,
            True,
            "info",
            f"{pk} unique across {df.height:,} rows",
        )

    example = (
        df.group_by(pk)
        .agg(
            pl.len().alias("n")
        )
        .filter(
            pl.col("n") > 1
        )
        .sort(
            "n",
            descending=True,
        )
        .head(3)
    )

    return Check(
        "primary_key",
        spec.name,
        False,
        "error",
        f"{dupes:,} duplicate rows under {pk}",
        {
            "duplicates": dupes,
            "worst": example.to_dicts(),
        },
    )


def check_primary_key(
    df: pl.DataFrame,
    spec: DatasetSpec,
) -> Check:
    """Validate primary-key uniqueness.

    Datasets with schema epochs are checked independently for every season using
    the contract that was actually valid in that season. This is essential for
    sources such as depth charts, whose 2025 feed has a completely different
    key from the 2001-2024 feed.
    """
    if not spec.schema_epochs:
        return _global_primary_key_check(
            df,
            spec,
        )

    seasons, schema_error = _schema_seasons(
        df,
        spec,
    )

    if schema_error is not None:
        return Check(
            "primary_key",
            spec.name,
            False,
            schema_error.severity,
            schema_error.detail,
            schema_error.data,
        )

    total_duplicates = 0
    failures: list[dict[str, Any]] = []

    for season in seasons:
        contract = spec.schema_for_season(
            season
        )

        pk = list(
            contract.primary_key
        )

        if not pk:
            continue

        season_df = _season_frame(
            df,
            season,
        )

        absent = [
            column
            for column in pk
            if column not in season_df.columns
        ]

        if absent:
            failures.append(
                {
                    "season": season,
                    "problem": "missing_columns",
                    "declared": pk,
                    "absent": absent,
                }
            )
            continue

        n_unique = (
            season_df.select(pk)
            .unique()
            .height
        )

        duplicates = (
            season_df.height
            - n_unique
        )

        if duplicates == 0:
            continue

        total_duplicates += duplicates

        example = (
            season_df.group_by(pk)
            .agg(
                pl.len().alias("n")
            )
            .filter(
                pl.col("n") > 1
            )
            .sort(
                "n",
                descending=True,
            )
            .head(3)
        )

        failures.append(
            {
                "season": season,
                "problem": "duplicates",
                "primary_key": pk,
                "duplicates": duplicates,
                "worst": example.to_dicts(),
            }
        )

    if failures:
        missing_failures = [
            failure
            for failure in failures
            if failure["problem"] == "missing_columns"
        ]

        if missing_failures:
            first = missing_failures[0]

            return Check(
                "primary_key",
                spec.name,
                False,
                "error",
                (
                    f"schema-aware key columns are absent in season "
                    f"{first['season']}: {first['absent']}"
                ),
                {
                    "failures": failures,
                },
            )

        return Check(
            "primary_key",
            spec.name,
            False,
            "error",
            (
                f"{total_duplicates:,} duplicate row(s) across "
                f"{len(failures)} schema-aware season(s)"
            ),
            {
                "duplicates": total_duplicates,
                "failures": failures,
            },
        )

    return Check(
        "primary_key",
        spec.name,
        True,
        "info",
        (
            f"schema-aware primary keys unique across "
            f"{df.height:,} rows and {len(seasons)} season(s)"
        ),
        {
            "seasons": seasons,
        },
    )


def _global_key_nulls(
    rates: dict[str, float],
    spec: DatasetSpec,
) -> dict[str, float]:
    return {
        column: round(
            rates[column],
            4,
        )
        for column in spec.primary_key
        if rates.get(
            column,
            0,
        )
        > 0
    }


def _schema_key_nulls(
    df: pl.DataFrame,
    spec: DatasetSpec,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Find null key values using the contract active in each season."""
    seasons, schema_error = _schema_seasons(
        df,
        spec,
    )

    if schema_error is not None:
        return (
            {},
            [
                {
                    "problem": "schema_contract",
                    "detail": schema_error.detail,
                    **schema_error.data,
                }
            ],
        )

    key_nulls: dict[str, float] = {}
    failures: list[dict[str, Any]] = []

    for season in seasons:
        contract = spec.schema_for_season(
            season
        )

        pk = list(
            contract.primary_key
        )

        if not pk:
            continue

        season_df = _season_frame(
            df,
            season,
        )

        absent = [
            column
            for column in pk
            if column not in season_df.columns
        ]

        if absent:
            failures.append(
                {
                    "season": season,
                    "problem": "missing_columns",
                    "absent": absent,
                    "declared": pk,
                }
            )
            continue

        if season_df.height == 0:
            continue

        for column in pk:
            null_rows = season_df.filter(
                pl.col(column).is_null()
            )

            if null_rows.height == 0:
                continue

            rules = [
                rule
                for rule in contract.nullable_key_rules
                if rule.column == column
            ]

            if rules:
                condition_fields = {
                    field_name
                    for rule in rules
                    for field_name, _expected in rule.when
                }

                missing_condition_fields = sorted(
                    field_name
                    for field_name in condition_fields
                    if field_name not in season_df.columns
                )

                if missing_condition_fields:
                    failures.append(
                        {
                            "season": season,
                            "problem": "missing_nullable_rule_columns",
                            "column": column,
                            "absent": missing_condition_fields,
                        }
                    )
                    continue

                allowed = pl.lit(False)

                for rule in rules:
                    matches = pl.lit(True)

                    for field_name, expected in rule.when:
                        condition = (
                            pl.col(field_name).is_null()
                            if expected is None
                            else pl.col(field_name) == pl.lit(expected)
                        )

                        matches = matches & condition

                    allowed = allowed | matches

                invalid_count = null_rows.filter(
                    ~allowed
                ).height
            else:
                invalid_count = null_rows.height

            if invalid_count == 0:
                continue

            rate = (
                invalid_count
                / season_df.height
            )

            key_nulls[
                f"{season}:{column}"
            ] = round(
                rate,
                4,
            )

    return (
        key_nulls,
        failures,
    )


def check_null_rates(
    df: pl.DataFrame,
    spec: DatasetSpec,
) -> Check:
    """Check sparse columns and null values in active primary keys.

    Ordinary feature sparsity is measured across the combined table. Primary
    key nulls are different: for schema-versioned datasets they are measured
    only against the key active for that row's season.
    """
    cols = _feature_columns(
        df
    )

    if (
        not cols
        or df.height == 0
    ):
        return Check(
            "null_rates",
            spec.name,
            True,
            "info",
            "nothing to measure",
        )

    nulls = (
        df.select(cols)
        .null_count()
        .row(0)
    )

    rates = {
        column: count / df.height
        for column, count in zip(
            cols,
            nulls,
        )
    }

    near_empty = {
        column: round(
            rate,
            4,
        )
        for column, rate in rates.items()
        if rate >= NEAR_EMPTY_THRESHOLD
    }

    if spec.schema_epochs:
        key_nulls, schema_failures = _schema_key_nulls(
            df,
            spec,
        )

        if schema_failures:
            return Check(
                "null_rates",
                spec.name,
                False,
                "error",
                "could not validate schema-aware primary-key nulls",
                {
                    "schema_failures": schema_failures,
                    "near_empty": near_empty,
                },
            )
    else:
        key_nulls = _global_key_nulls(
            rates,
            spec,
        )

    if key_nulls:
        return Check(
            "null_rates",
            spec.name,
            False,
            "error",
            f"primary key columns contain nulls: {key_nulls}",
            {
                "key_nulls": key_nulls,
                "near_empty": near_empty,
            },
        )

    if near_empty:
        return Check(
            "null_rates",
            spec.name,
            True,
            "warning",
            (
                f"{len(near_empty)} columns are "
                f">={int(NEAR_EMPTY_THRESHOLD * 100)}% null -- "
                "treat as 'not collected', do not impute: "
                f"{sorted(near_empty)[:8]}"
            ),
            {
                "near_empty": near_empty,
            },
        )

    return Check(
        "null_rates",
        spec.name,
        True,
        "info",
        f"{len(cols)} columns, no near-empty ones",
    )


def check_row_counts(
    df: pl.DataFrame,
    spec: DatasetSpec,
) -> Check:
    if "season" not in df.columns:
        return Check(
            "row_counts",
            spec.name,
            True,
            "info",
            "not season-partitioned",
        )

    counts = (
        df.group_by("season")
        .agg(
            pl.len().alias("rows")
        )
        .sort("season")
    )

    rows = counts.to_dicts()
    anomalies: list[dict[str, Any]] = []

    for previous, current in itertools.pairwise(rows):
        if previous["rows"] == 0:
            continue

        delta = (
            abs(
                current["rows"]
                - previous["rows"]
            )
            / previous["rows"]
        )

        if delta > ROW_COUNT_TOLERANCE:
            anomalies.append(
                {
                    "from": previous["season"],
                    "to": current["season"],
                    "from_rows": previous["rows"],
                    "to_rows": current["rows"],
                    "change": round(
                        delta,
                        3,
                    ),
                }
            )

    if anomalies:
        return Check(
            "row_counts",
            spec.name,
            True,
            "warning",
            (
                f"{len(anomalies)} season-over-season row-count jumps "
                f">{int(ROW_COUNT_TOLERANCE * 100)}%; check for a partial "
                "download or a genuine format change"
            ),
            {
                "anomalies": anomalies,
            },
        )

    return Check(
        "row_counts",
        spec.name,
        True,
        "info",
        f"{len(rows)} seasons, stable volumes",
    )


def check_season_contiguity(
    df: pl.DataFrame,
    spec: DatasetSpec,
) -> Check:
    if (
        "season" not in df.columns
        or df.height == 0
    ):
        return Check(
            "contiguity",
            spec.name,
            True,
            "info",
            "not applicable",
        )

    present = _seasons_present(
        df
    )

    holes = sorted(
        set(
            range(
                present[0],
                present[-1] + 1,
            )
        )
        - set(present)
    )

    if holes:
        return Check(
            "contiguity",
            spec.name,
            False,
            "error",
            f"missing seasons inside the ingested range: {holes}",
            {
                "present": present,
                "holes": holes,
            },
        )

    return Check(
        "contiguity",
        spec.name,
        True,
        "info",
        f"contiguous {present[0]}-{present[-1]}",
    )


def check_target_coverage(
    df: pl.DataFrame,
    spec: DatasetSpec,
) -> Check:
    cfg = get_settings()

    if "season" not in df.columns:
        return Check(
            "target_coverage",
            spec.name,
            True,
            "info",
            "not season-partitioned",
        )

    present = set(
        _seasons_present(
            df
        )
    )

    gap = sorted(
        set(cfg.target_seasons)
        - present
    )

    if not gap:
        return Check(
            "target_coverage",
            spec.name,
            True,
            "info",
            "full target window present",
        )

    severity = "warning"

    return Check(
        "target_coverage",
        spec.name,
        True,
        severity,
        (
            f"{len(gap)} target seasons absent: {gap}. "
            "This is coverage, not corruption -- features from this dataset "
            "must be null there."
        ),
        {
            "missing_seasons": gap,
        },
    )


def check_referential_integrity(
    df: pl.DataFrame,
    spec: DatasetSpec,
) -> Check:
    """Child tables must reference games that exist in the canonical game table."""
    game_cols = [
        column
        for column in (
            "game_id",
            "nflverse_game_id",
        )
        if column in df.columns
    ]

    if (
        not game_cols
        or spec.name == "schedules"
    ):
        return Check(
            "referential_integrity",
            spec.name,
            True,
            "info",
            "no game reference",
        )

    column = game_cols[0]

    try:
        games = read_dataset(
            get_manifest()["schedules"],
            root=RAW_DIR,
        )
    except FileNotFoundError:
        return Check(
            "referential_integrity",
            spec.name,
            True,
            "warning",
            "schedules not ingested; cannot check game references",
        )

    known = set(
        games.get_column(
            "game_id"
        ).to_list()
    )

    referenced = set(
        df.get_column(column)
        .drop_nulls()
        .to_list()
    )

    orphans = sorted(
        referenced
        - known
    )

    if orphans:
        return Check(
            "referential_integrity",
            spec.name,
            False,
            "error",
            (
                f"{len(orphans)} game ids not present in schedules "
                f"(e.g. {orphans[:3]})"
            ),
            {
                "orphan_count": len(orphans),
                "examples": orphans[:10],
            },
        )

    return Check(
        "referential_integrity",
        spec.name,
        True,
        "info",
        (
            f"all {len(referenced):,} referenced games "
            "exist in schedules"
        ),
    )


CHECKS = (
    check_primary_key,
    check_null_rates,
    check_row_counts,
    check_season_contiguity,
    check_target_coverage,
    check_referential_integrity,
)


def validate_dataset(
    name: str,
) -> list[Check]:
    spec = get_manifest()[name]

    try:
        df = read_dataset(
            spec,
            root=RAW_DIR,
        )
    except FileNotFoundError as exc:
        severity = (
            "error"
            if spec.required
            else "info"
        )

        return [
            Check(
                "present",
                name,
                not spec.required,
                severity,
                str(exc),
            )
        ]

    results = [
        Check(
            "present",
            name,
            True,
            "info",
            f"{df.height:,} rows, {df.width} columns",
        )
    ]

    for check in CHECKS:
        try:
            results.append(
                check(
                    df,
                    spec,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep independent checks running
            results.append(
                Check(
                    check.__name__,
                    name,
                    False,
                    "error",
                    f"check crashed: {exc}",
                )
            )

    return results


def validate_all(
    names: list[str] | None = None,
) -> dict[str, Any]:
    mf = get_manifest()

    names = (
        names
        or mf.names()
    )

    all_checks: list[Check] = []

    for name in names:
        log.info(
            "validating %s",
            name,
        )

        all_checks.extend(
            validate_dataset(
                name
            )
        )

    errors = [
        check
        for check in all_checks
        if (
            not check.passed
            and check.severity == "error"
        )
    ]

    warnings = [
        check
        for check in all_checks
        if check.severity == "warning"
    ]

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "datasets_checked": names,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "checks": [
            asdict(check)
            for check in all_checks
        ],
    }

    QUALITY_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    QUALITY_REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return report


def format_report(
    report: dict[str, Any],
) -> str:
    lines = [
        (
            f"quality report {report['generated_at'][:19]}Z -- "
            f"{report['n_errors']} error(s), "
            f"{report['n_warnings']} warning(s)"
        ),
        "-" * 78,
    ]

    by_dataset: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for check in report["checks"]:
        by_dataset.setdefault(
            check["dataset"],
            [],
        ).append(
            check
        )

    for dataset, checks in sorted(
        by_dataset.items()
    ):
        if any(
            (
                not check["passed"]
                and check["severity"] == "error"
            )
            for check in checks
        ):
            worst = "ERROR"
        elif any(
            check["severity"] == "warning"
            for check in checks
        ):
            worst = "WARN"
        else:
            worst = "ok"

        lines.append(
            f"[{worst:>5}] {dataset}"
        )

        for check in checks:
            if (
                check["severity"] == "info"
                and check["passed"]
                and check["name"] != "present"
            ):
                continue

            lines.append(
                f"         {check['name']}: "
                f"{check['detail']}"
            )

    return "\n".join(
        lines
    )