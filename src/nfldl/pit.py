"""Point-in-time discipline: the module that stops the future leaking backwards.

The brief's hardest requirement is a single sentence -- *never use postgame
injury knowledge in a pregame historical prediction* -- and it generalises to
almost every table here. Snap counts, weekly stats, NGS and PFR advanced stats
are all **postgame**: the week-W row for a game only exists because the game was
played. Joining them to that same game is not a subtle bug, it is a model that
knows the answer.

The primitives below make the safe thing the easy thing:

``knowledge_time``   when a fact became knowable, per dataset semantics.
``asof_join``        strict as-of join with an explicit cutoff.
``lag_weekly``       shift week-grained postgame data forward by N weeks.
``audit_frame``      scan a feature frame for columns that cannot be pregame.

Everything here is timestamp-first. Where a dataset only has week granularity,
we resolve it to a timestamp via the schedule, because "week 5" is ambiguous
across a Thursday game and the following Monday.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Literal

import polars as pl

log = logging.getLogger(__name__)

Availability = Literal["pregame", "postgame", "static", "unknown"]

# Per-dataset semantics. This is the codified version of the manifest's
# `leakage_risk` field, at column granularity where it matters.
DATASET_AVAILABILITY: dict[str, Availability] = {
    "schedules": "pregame",       # except result/score columns -- see POSTGAME_COLUMNS
    "teams": "static",
    "players": "static",
    "draft_picks": "static",
    "combine": "static",
    "rosters": "pregame",
    "weekly_rosters": "pregame",
    "depth_charts": "pregame",
    "injuries": "pregame",        # practice + game-status reports precede kickoff
    "officials": "pregame",       # crews are announced before the game
    "stadiums": "static",
    "coaching_staff": "pregame",
    "trades": "pregame",
    "contracts": "unknown",       # current-state table; see manifest caveat
    "play_by_play": "postgame",
    "player_stats": "postgame",
    "team_stats": "postgame",
    "snap_counts": "postgame",
    "participation": "postgame",
    "ftn_charting": "postgame",
    "nextgen_stats": "postgame",
    "pfr_advstats": "postgame",
    "weather": "unknown",         # ERA5 reanalysis is not a pregame forecast
}

# Columns that are postgame even inside an otherwise-pregame table.
POSTGAME_COLUMNS: frozenset[str] = frozenset({
    "result", "total", "overtime", "home_score", "away_score",
    "temp", "wind",  # gamebook-observed, not forecast
})

# Betting markets are pregame *if* you use the closing line honestly. They are
# not "free" though: they encode information a pure fundamentals model is
# claiming to derive. Flagged so the choice is explicit.
MARKET_COLUMNS: frozenset[str] = frozenset({
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "away_spread_odds", "home_spread_odds", "under_odds", "over_odds",
})


class LeakageError(AssertionError):
    """Raised when a frame provably contains information from after the cutoff."""


@dataclass(frozen=True)
class LeakageFinding:
    column: str
    reason: str
    severity: Literal["error", "warning"]


def knowledge_time(
    df: pl.DataFrame,
    dataset: str,
    *,
    kickoff_col: str = "kickoff_utc",
    report_col: str | None = None,
    postgame_delay_hours: float = 4.0,
) -> pl.DataFrame:
    """Attach ``_known_at``: the earliest instant the row's content was knowable.

    * ``static``   -> far past; always available.
    * ``pregame``  -> the report timestamp if the table has one, else kickoff.
    * ``postgame`` -> kickoff plus a delay covering the game itself.
    * ``unknown``  -> treated as postgame. Unknown provenance is not a licence.
    """
    availability = DATASET_AVAILABILITY.get(dataset, "unknown")

    if availability == "static":
        return df.with_columns(
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("_known_at"),
            pl.lit("static").alias("_availability"),
        )

    if kickoff_col not in df.columns:
        raise ValueError(
            f"{dataset}: cannot compute knowledge time without '{kickoff_col}'. "
            "Join the schedule first -- week numbers alone are not timestamps."
        )

    if availability == "pregame":
        if report_col and report_col in df.columns:
            known = pl.min_horizontal(
                pl.col(report_col).cast(pl.Datetime("us", "UTC")), pl.col(kickoff_col)
            )
        else:
            known = pl.col(kickoff_col)
    else:
        known = pl.col(kickoff_col) + pl.duration(hours=postgame_delay_hours)

    return df.with_columns(
        known.alias("_known_at"), pl.lit(availability).alias("_availability")
    )


def asof_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    cutoff_col: str = "kickoff_utc",
    known_col: str = "_known_at",
    by: Iterable[str] | None = None,
    tolerance: timedelta | None = None,
    suffix: str = "_hist",
) -> pl.DataFrame:
    """As-of join that only ever looks backwards, strictly.

    ``left`` supplies the prediction instants (usually kickoffs). ``right`` must
    carry ``_known_at``. Rows in ``right`` whose ``_known_at`` is not strictly
    before the cutoff cannot be selected -- including exact ties, which are the
    classic off-by-one that lets a game's own result in.
    """
    if known_col not in right.columns:
        raise ValueError(f"right frame needs '{known_col}'; call knowledge_time() first")
    if cutoff_col not in left.columns:
        raise ValueError(f"left frame needs '{cutoff_col}'")

    by = list(by or [])
    # Nudge back by one microsecond so join_asof's inclusive `<=` becomes `<`.
    right_shift = right.with_columns(
        (pl.col(known_col) + pl.duration(microseconds=1)).alias("_asof_key")
    ).sort([*by, "_asof_key"])
    left_sorted = left.sort([*by, cutoff_col])

    # We sorted both frames above; polars cannot verify that through `by`
    # groups, and its advisory warning here is noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns")
        return left_sorted.join_asof(
            right_shift,
            left_on=cutoff_col,
            right_on="_asof_key",
            by=by or None,
            strategy="backward",
            tolerance=tolerance,
            suffix=suffix,
        ).drop("_asof_key", strict=False)


def lag_weekly(
    df: pl.DataFrame,
    value_cols: Iterable[str],
    *,
    by: Iterable[str] = ("season", "team"),
    weeks: int = 1,
) -> pl.DataFrame:
    """Shift postgame weekly measures forward so week W sees only weeks < W.

    Sorting by week within season is correct here because these tables are
    week-grained by construction; the tie-break on ``week`` is stable.
    """
    by = list(by)
    if "week" not in df.columns:
        raise ValueError("lag_weekly needs a 'week' column")
    ordered = df.sort([*by, "week"])
    return ordered.with_columns(
        [pl.col(c).shift(weeks).over(by).alias(f"{c}_lag{weeks}") for c in value_cols]
    )


def audit_frame(
    df: pl.DataFrame,
    *,
    dataset_of: dict[str, str] | None = None,
    allow_market: bool = False,
    raise_on_error: bool = True,
) -> list[LeakageFinding]:
    """Scan a modelling frame for columns that cannot be known before kickoff.

    Catches the ordinary mistakes: raw postgame columns carried through, an
    outcome column left in the feature set, an unlagged postgame measure, and
    the ingest bookkeeping columns being treated as features.
    """
    findings: list[LeakageFinding] = []
    dataset_of = dataset_of or {}

    for col in df.columns:
        if col in POSTGAME_COLUMNS:
            findings.append(LeakageFinding(col, "outcome column from the game being predicted", "error"))
            continue

        if col in MARKET_COLUMNS and not allow_market:
            findings.append(
                LeakageFinding(col, "betting market column; pregame but encodes others' information", "warning")
            )
            continue

        if col.startswith("_ingested_at") or col in ("_source_tier", "_provider"):
            findings.append(
                LeakageFinding(col, "ingest bookkeeping; processing time is not knowledge time", "warning")
            )
            continue

        origin = dataset_of.get(col)
        if origin and DATASET_AVAILABILITY.get(origin) == "postgame":
            if not any(tag in col for tag in ("_lag", "_prior", "_rolling", "_ewma", "_todate")):
                findings.append(
                    LeakageFinding(
                        col,
                        f"sourced from postgame dataset '{origin}' with no lag marker in the name",
                        "error",
                    )
                )

    errors = [f for f in findings if f.severity == "error"]
    if errors and raise_on_error:
        detail = "\n".join(f"  - {f.column}: {f.reason}" for f in errors)
        raise LeakageError(f"{len(errors)} leaking column(s):\n{detail}")

    for f in findings:
        log.warning("leakage %s: %s (%s)", f.severity, f.column, f.reason)
    return findings


def assert_no_future_rows(
    df: pl.DataFrame, cutoff: pl.Expr | str = "kickoff_utc", known_col: str = "_known_at"
) -> None:
    """Hard check that no joined row was knowable only after its cutoff."""
    cutoff_expr = pl.col(cutoff) if isinstance(cutoff, str) else cutoff
    if known_col not in df.columns:
        return
    bad = df.filter(pl.col(known_col).is_not_null() & (pl.col(known_col) >= cutoff_expr))
    if bad.height:
        raise LeakageError(
            f"{bad.height} rows carry information first knowable at or after their own cutoff. "
            "Use pit.asof_join rather than a plain join."
        )
