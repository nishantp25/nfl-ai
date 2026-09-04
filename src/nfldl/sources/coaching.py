"""Optional curated non-head-coach staff history.

Head-coach identity belongs to the canonical nflverse schedules table:
``home_coach`` and ``away_coach`` identify the HC assigned to each NFL game.
Those game-level fields must be used directly rather than converted into
invented employment-tenure dates. Future schedules may already contain coach
assignments, so a future scheduled game is not evidence that a coach's tenure
is known to continue through that date.

This module therefore stores only sourced non-HC staff tenures such as OC, DC,
STC and PGC from ``configs/coaching_staff.csv``. Every curated row requires
provenance.

Coaching tendencies are deliberately not stored here. They are derived from
prior play-by-play so historical features can be reconstructed point-in-time
without leaking later games or seasons.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import polars as pl

from ..config import CONFIG_DIR

log = logging.getLogger(__name__)

CURATED_CSV = CONFIG_DIR / "coaching_staff.csv"
CURATED_COLUMNS = [
    "coach_name", "team", "role", "start_date", "end_date",
    "interim", "source_url", "source_note",
]
VALID_ROLES = {"OC", "DC", "STC", "PGC"}


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


def load_curated() -> pl.DataFrame:
    """Coordinator rows. Returns an empty, correctly-typed frame if absent."""
    schema = {
        "coach_name": pl.String, "team": pl.String, "role": pl.String,
        "start_date": pl.Date, "end_date": pl.Date, "interim": pl.Boolean,
        "source_url": pl.String, "source_note": pl.String,
    }
    if not CURATED_CSV.exists():
        log.warning(
            "no curated staff file at %s -- coaching_staff will be empty. "
            "Coordinator/staff features will be unavailable, not silently wrong.",
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

    roles = set(
        df.get_column("role").unique().to_list()
    )

    if "HC" in roles:
        raise ValueError(
            "head coach rows are not allowed in coaching_staff; "
            "game-level HC identity belongs to nflverse schedules"
        )

    bad_roles = roles - VALID_ROLES
    if bad_roles:
        raise ValueError(
            f"unknown coaching roles in {CURATED_CSV.name}: "
            f"{sorted(bad_roles)}"
        )

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
    """Build the optional curated non-head-coach staff dimension.

    Head-coach identity deliberately does not live in this table. nflverse
    schedules already provide the HC assigned to each game, which is the
    correct game-level source for point-in-time modelling.
    """
    curated = load_curated()

    if curated.is_empty():
        return curated.with_columns(
            pl.lit(None, dtype=pl.String).alias("coach_id"),
            pl.lit(None, dtype=pl.String).alias("source_tier"),
        ).select(
            "coach_id",
            "coach_name",
            "team",
            "role",
            "start_date",
            "end_date",
            "interim",
            "source_tier",
            "source_note",
            "source_url",
        )

    out = (
        curated.with_columns(
            coach_id_expr().alias("coach_id"),
            pl.lit("curated_fallback").alias("source_tier"),
        )
        .select(
            "coach_id",
            "coach_name",
            "team",
            "role",
            "start_date",
            "end_date",
            "interim",
            "source_tier",
            "source_note",
            "source_url",
        )
        .sort(
            [
                "team",
                "role",
                "start_date",
            ]
        )
    )

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
