"""Coaching staff history: identity and tenure only.

Head coaches do not need to be curated at all. ``nflverse`` schedules carry
``home_coach`` / ``away_coach`` on every game, which means head-coach tenure is
*derivable from the canonical game table* -- including mid-season firings and
interim appointments, dated to the exact game. That is strictly better than a
hand-maintained list: it cannot drift, and it is reproducible.

Coordinators (OC/DC) are not in nflverse. Those come from
``configs/coaching_staff.csv``, curated from official team and league
announcements, with ``source_url`` mandatory on every row.

What this module deliberately does NOT store: tendencies. No "aggressive on
fourth down", no scheme labels, no pace descriptors. Those are derived from
play-by-play, per the manifest's ``forbidden_downloads`` list, so they can be
recomputed as of any historical date without leaking later seasons.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import polars as pl

from ..config import CONFIG_DIR, RAW_DIR
from ..manifest import get_manifest

log = logging.getLogger(__name__)

CURATED_CSV = CONFIG_DIR / "coaching_staff.csv"
CURATED_COLUMNS = [
    "coach_name", "team", "role", "start_date", "end_date",
    "interim", "source_url", "source_note",
]
VALID_ROLES = {"HC", "OC", "DC", "STC", "PGC"}


def _as_date(col: str) -> pl.Expr:
    """Parse gameday whether nflverse ships it as Date or as a string."""
    return (
        pl.when(pl.col(col).cast(pl.String).is_not_null())
        .then(pl.col(col).cast(pl.String).str.to_date("%Y-%m-%d", strict=False))
        .otherwise(None)
        .alias(col)
    )


def coach_id(name: str) -> str:
    """Stable slug for a coach. Names are the only key nflverse gives us.

    This is a name-based key, which the brief rightly warns against. It is
    unavoidable here -- there is no coach ID anywhere in the public data -- so
    it is confined to this one dimension and never used to join player data.
    """
    out = name.strip().lower()
    for ch in ".'`,":
        out = out.replace(ch, "")
    return out.replace(" ", "_").replace("-", "_")


def coach_id_expr(col: str = "coach_name") -> pl.Expr:
    """Vectorised twin of :func:`coach_id`. Keep the two in sync."""
    e = pl.col(col).str.strip_chars().str.to_lowercase()
    for ch in ".'`,":
        e = e.str.replace_all(ch, "", literal=True)
    return e.str.replace_all(" ", "_", literal=True).str.replace_all("-", "_", literal=True)


def head_coach_stints(schedules: pl.DataFrame) -> pl.DataFrame:
    """Derive HC tenures by run-length encoding coach-per-game-date per team.

    A "stint" is a maximal consecutive run of games a team played under one
    coach. Splitting on discontinuity rather than on season means a mid-season
    change produces two correctly-dated rows, and a coach who returns after a
    gap produces two stints rather than one wrongly-continuous one.
    """
    needed = {"season", "gameday", "home_team", "away_team", "home_coach", "away_coach"}
    missing = needed - set(schedules.columns)
    if missing:
        raise ValueError(f"schedules is missing {sorted(missing)}; cannot derive coaches")

    home = schedules.select(
        pl.col("season"),
        _as_date("gameday"),
        pl.col("game_id"),
        pl.col("home_team").alias("team"),
        pl.col("home_coach").alias("coach_name"),
    )
    away = schedules.select(
        pl.col("season"),
        _as_date("gameday"),
        pl.col("game_id"),
        pl.col("away_team").alias("team"),
        pl.col("away_coach").alias("coach_name"),
    )

    long = (
        pl.concat([home, away])
        .drop_nulls(["coach_name", "gameday"])
        .filter(pl.col("coach_name").str.strip_chars() != "")
        .sort(["team", "gameday"])
    )

    # New stint whenever the coach changes for a team, in date order.
    long = long.with_columns(
        (pl.col("coach_name") != pl.col("coach_name").shift(1).over("team"))
        .fill_null(True)
        .cum_sum()
        .over("team")
        .alias("stint_no")
    )

    stints = (
        long.group_by(["team", "stint_no"])
        .agg(
            pl.col("coach_name").first(),
            pl.col("gameday").min().alias("first_game"),
            pl.col("gameday").max().alias("last_game"),
            pl.col("season").min().alias("first_season"),
            pl.col("season").max().alias("last_season"),
            pl.len().alias("games"),
        )
        .sort(["team", "first_game"])
    )

    return stints.select(
        coach_id_expr().alias("coach_id"),
        "coach_name",
        "team",
        pl.lit("HC").alias("role"),
        pl.col("first_game").alias("start_date"),
        pl.col("last_game").alias("end_date"),
        "first_season",
        "last_season",
        "games",
        pl.lit(False).alias("interim"),
        pl.lit("nflverse").alias("source_tier"),
        pl.lit("derived from nflverse schedules home_coach/away_coach").alias("source_note"),
        pl.lit("https://github.com/nflverse/nflverse-data/releases/tag/schedules").alias("source_url"),
    )


def load_curated() -> pl.DataFrame:
    """Coordinator rows. Returns an empty, correctly-typed frame if absent."""
    schema = {
        "coach_name": pl.String, "team": pl.String, "role": pl.String,
        "start_date": pl.Date, "end_date": pl.Date, "interim": pl.Boolean,
        "source_url": pl.String, "source_note": pl.String,
    }
    if not CURATED_CSV.exists():
        log.warning(
            "no coordinator file at %s -- coaching_staff will contain head coaches only. "
            "OC/DC features will be unavailable, not silently wrong.",
            CURATED_CSV,
        )
        return pl.DataFrame(schema=schema)

    df = pl.read_csv(CURATED_CSV, schema_overrides={"start_date": pl.String, "end_date": pl.String})
    missing = [c for c in CURATED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{CURATED_CSV.name} missing columns: {missing}")

    df = df.with_columns(
        pl.col("start_date").str.strptime(pl.Date, "%Y-%m-%d", strict=True),
        pl.col("end_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("interim").cast(pl.Boolean),
    )

    bad_roles = set(df.get_column("role").unique().to_list()) - VALID_ROLES
    if bad_roles:
        raise ValueError(f"unknown coaching roles in {CURATED_CSV.name}: {sorted(bad_roles)}")

    unsourced = df.filter(
        pl.col("source_url").is_null() | (pl.col("source_url").str.strip_chars() == "")
    )
    if unsourced.height:
        raise ValueError(
            f"{unsourced.height} curated coaching rows lack source_url. A curated fallback row "
            "without provenance is not admissible."
        )
    return df


def build(**_: Any) -> pl.DataFrame:
    """Union derived head-coach stints with curated coordinator rows."""
    from ..storage import read_dataset

    mf = get_manifest()
    sched = read_dataset(mf["schedules"], root=RAW_DIR)
    hc = head_coach_stints(sched)

    curated = load_curated()
    if curated.height:
        curated = curated.with_columns(
            coach_id_expr().alias("coach_id"),
            pl.col("start_date").dt.year().alias("first_season"),
            pl.col("end_date").dt.year().alias("last_season"),
            pl.lit(None, dtype=pl.UInt32).alias("games"),
            pl.lit("curated_fallback").alias("source_tier"),
        ).select(hc.columns)
        out = pl.concat([hc, curated], how="vertical_relaxed")
    else:
        out = hc

    out = out.sort(["team", "role", "start_date"])
    _warn_on_gaps(out)
    return out


def _warn_on_gaps(staff: pl.DataFrame) -> None:
    """Flag teams with no coordinator coverage, so absence is visible."""
    have_coord = set(
        staff.filter(pl.col("role").is_in(["OC", "DC"])).get_column("team").unique().to_list()
    )
    all_teams = set(staff.get_column("team").unique().to_list())
    missing = sorted(all_teams - have_coord)
    if missing:
        log.warning(
            "no OC/DC rows for %d teams (%s...). Coordinator features must be null for these, "
            "never imputed.",
            len(missing), ", ".join(missing[:8]),
        )


def staff_as_of(staff: pl.DataFrame, team: str, role: str, when: date) -> pl.DataFrame:
    """Who held `role` for `team` on `when`. Point-in-time by construction."""
    return staff.filter(
        (pl.col("team") == team)
        & (pl.col("role") == role)
        & (pl.col("start_date") <= when)
        & (pl.col("end_date").is_null() | (pl.col("end_date") >= when))
    )
