"""Stadium dimension.

nflverse schedules already carry per-game ``roof`` and ``surface``, which is the
*correct* representation: it handles the year a team played somewhere else, the
week the roof was closed, and the season a field was re-turfed. So schedules is
the primary source and we do not second-guess it.

What schedules does not carry is geography. Coordinates, elevation and timezone
come from a curated CSV (``configs/stadiums.csv``) held to the manifest's
``curated_fallback`` tier, with a ``source_note`` on every row.

Rows are slowly-changing-dimension type 2: ``valid_from`` / ``valid_to``, because
teams relocate and venues are rebuilt. Joining a 2015 game must not pick up 2020
geometry.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from ..config import CONFIG_DIR, RAW_DIR
from ..manifest import get_manifest

log = logging.getLogger(__name__)

CURATED_CSV = CONFIG_DIR / "stadiums.csv"

CURATED_SCHEMA = {
    "stadium_id": pl.String,
    "stadium_name": pl.String,
    "city": pl.String,
    "state": pl.String,
    "country": pl.String,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "elevation_m": pl.Float64,
    "timezone": pl.String,
    "default_roof": pl.String,
    "default_surface": pl.String,
    "valid_from": pl.Date,
    "valid_to": pl.Date,
    "source_url": pl.String,
    "source_note": pl.String,
}


def load_curated() -> pl.DataFrame:
    if not CURATED_CSV.exists():
        raise FileNotFoundError(
            f"curated stadium table missing at {CURATED_CSV}. It is a required fallback "
            "source; see configs/data_sources.yaml -> stadiums."
        )
    df = pl.read_csv(CURATED_CSV, schema_overrides={"valid_from": pl.String, "valid_to": pl.String})

    missing = [c for c in CURATED_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"{CURATED_CSV.name} is missing columns: {missing}")

    df = df.with_columns(
        pl.col("valid_from").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("valid_to").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
    )

    # Provenance is not optional for a curated fallback.
    unsourced = df.filter(
        pl.col("source_url").is_null() | (pl.col("source_url").str.strip_chars() == "")
    )
    if unsourced.height:
        raise ValueError(
            f"{unsourced.height} curated stadium rows have no source_url. Every manually "
            "curated row must be attributable."
        )

    # Overlapping validity windows for one stadium would make the as-of join
    # ambiguous, which is exactly the bug SCD-2 exists to prevent.
    for sid, grp in df.sort(["stadium_id", "valid_from"]).group_by("stadium_id", maintain_order=True):
        rows = grp.to_dicts()
        for prev, nxt in zip(rows, rows[1:]):
            prev_end = prev["valid_to"] or date.max
            if nxt["valid_from"] <= prev_end and prev["valid_to"] is not None:
                if nxt["valid_from"] < prev_end:
                    raise ValueError(
                        f"stadium '{sid[0] if isinstance(sid, tuple) else sid}' has overlapping "
                        f"validity windows ending {prev['valid_to']} and starting {nxt['valid_from']}"
                    )
    return df


def build(**_: Any) -> pl.DataFrame:
    """Join per-game venue facts from schedules onto curated geography.

    Output is one row per (stadium_id, valid_from) plus the per-game roof and
    surface observations rolled up so that gaps in the curated file are visible
    rather than silently imputed.
    """
    from ..storage import read_dataset  # local import avoids a cycle

    mf = get_manifest()
    sched = read_dataset(mf["schedules"], root=RAW_DIR)

    have = [c for c in ("stadium_id", "stadium", "roof", "surface", "season", "gameday") if c in sched.columns]
    observed = sched.select(have)

    if "stadium_id" not in observed.columns:
        log.warning("schedules has no stadium_id column; falling back to stadium name as key")
        observed = observed.with_columns(pl.col("stadium").alias("stadium_id"))

    usage = (
        observed.drop_nulls("stadium_id")
        .group_by("stadium_id")
        .agg(
            pl.col("stadium").drop_nulls().mode().first().alias("schedule_stadium_name"),
            pl.col("season").min().alias("first_season_seen"),
            pl.col("season").max().alias("last_season_seen"),
            pl.len().alias("games_observed"),
            pl.col("roof").drop_nulls().unique().sort().alias("roof_values_seen"),
            pl.col("surface").drop_nulls().unique().sort().alias("surface_values_seen"),
        )
    )

    curated = load_curated()
    out = usage.join(curated, on="stadium_id", how="full", coalesce=True)

    out = out.with_columns(
        pl.col("latitude").is_not_null().alias("has_coordinates"),
        pl.when(pl.col("default_roof").is_in(["dome", "closed"]))
        .then(True)
        .otherwise(False)
        .alias("is_indoor_default"),
    )

    ungeocoded = out.filter(
        pl.col("games_observed").is_not_null() & ~pl.col("has_coordinates")
    )
    if ungeocoded.height:
        # Loud, not fatal: a weather join will skip these and say so.
        log.warning(
            "%d stadiums appear in schedules with no curated coordinates: %s",
            ungeocoded.height,
            ungeocoded.get_column("stadium_id").to_list()[:20],
        )

    return out.sort("stadium_id")


def geography_for_games(games: pl.DataFrame, stadiums: pl.DataFrame) -> pl.DataFrame:
    """Attach point-in-time venue geography to a game table.

    Picks the curated row whose validity window contains ``gameday`` -- so a
    2015 Rams home game gets St. Louis, not Los Angeles.
    """
    if "gameday" not in games.columns:
        raise ValueError("games frame needs a gameday column")

    geo = stadiums.select(
        "stadium_id", "latitude", "longitude", "elevation_m", "timezone",
        "valid_from", "valid_to", "default_roof",
    ).drop_nulls("latitude")

    joined = games.join(geo, on="stadium_id", how="left")
    gameday = pl.col("gameday").cast(pl.Date)
    return joined.filter(
        pl.col("latitude").is_null()
        | (
            (pl.col("valid_from").is_null() | (gameday >= pl.col("valid_from")))
            & (pl.col("valid_to").is_null() | (gameday <= pl.col("valid_to")))
        )
    )


def write_template(path: Path | None = None) -> Path:  # pragma: no cover - scaffolding
    """Emit a CSV skeleton listing every stadium_id seen in schedules."""
    from ..storage import read_dataset

    path = path or (CONFIG_DIR / "stadiums.template.csv")
    sched = read_dataset(get_manifest()["schedules"], root=RAW_DIR)
    ids = (
        sched.select("stadium_id", "stadium")
        .drop_nulls("stadium_id")
        .unique(subset=["stadium_id"])
        .sort("stadium_id")
    )
    skeleton = ids.with_columns(
        [pl.lit(None, dtype=dt).alias(col) for col, dt in CURATED_SCHEMA.items()
         if col not in ("stadium_id", "stadium_name")]
    ).rename({"stadium": "stadium_name"})
    skeleton.select(list(CURATED_SCHEMA)).write_csv(path)
    return path
