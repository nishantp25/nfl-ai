"""Historical weather at kickoff.

Provider: **Open-Meteo Historical Weather API** (ERA5 / ERA5T reanalysis).
Chosen because it is free, keyless, global (the NFL plays in London, Munich,
Mexico City and São Paulo), and covers 1940-present, so there is no coverage gap
anywhere in the 2011+ window. NOAA/NCEI and Meteostat are documented
alternatives in the manifest; NOAA needs a token and is US-only, and Meteostat's
station coverage is thin at some venues.

Three things this module is careful about:

**Roof.** A closed-roof or dome game does not get an outdoor observation. It
gets neutral values and ``is_indoor = true``. Feeding a 4°F outdoor reading into
a game played in Ford Field is a straightforward way to teach a model nonsense.
Retractable-roof venues are resolved per game from ``schedules.roof``, never
from a stadium-level default.

**Leakage.** ERA5 is a *reanalysis*: it is what the weather actually was, not
what anyone knew beforehand. For a pregame feature this is mildly optimistic.
The output carries ``weather_is_reanalysis = true`` so the choice stays visible
downstream, and ``nfldl.pit`` treats it as postgame-adjacent unless you
explicitly opt in.

**Batching.** One request per (venue, season) rather than per game, which is
roughly 40x fewer calls and stays comfortably inside the public rate limits.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import timedelta
from typing import Any, Iterable

import polars as pl

from ..config import RAW_DIR, get_settings
from ..http import get_json
from ..manifest import get_manifest

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "snowfall",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "cloud_cover",
]

INDOOR_ROOFS = {"dome", "closed"}

# Neutral indoor conditions. Not measurements -- a constant, flagged as such,
# so no model mistakes them for a signal.
INDOOR_DEFAULTS = {
    "temperature_c": 21.0,
    "relative_humidity": 50.0,
    "dew_point_c": 10.0,
    "apparent_temperature_c": 21.0,
    "precipitation_mm": 0.0,
    "snowfall_cm": 0.0,
    "wind_speed_kmh": 0.0,
    "wind_gust_kmh": 0.0,
    "wind_direction_deg": None,
    "cloud_cover_pct": 0.0,
}


def _fetch_venue_season(lat: float, lon: float, start: str, end: str) -> pl.DataFrame:
    """One archive call covering a venue's whole season."""
    payload = get_json(
        ARCHIVE_URL,
        params={
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
        },
    )
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        return pl.DataFrame()

    df = pl.DataFrame(hourly).with_columns(
        pl.col("time")
        .str.strptime(pl.Datetime("us"), "%Y-%m-%dT%H:%M", strict=False)
        .dt.replace_time_zone("UTC")
        .alias("obs_utc")
    )
    return df.drop("time").with_columns(
        pl.lit(round(lat, 4)).alias("_lat"), pl.lit(round(lon, 4)).alias("_lon")
    )


def _nearest_hour(games: pl.DataFrame, obs: pl.DataFrame) -> pl.DataFrame:
    """Join each kickoff to the closest hourly observation (±90 minutes)."""
    if obs.is_empty():
        return games.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(v) for v in HOURLY_VARS]
        )
    left = games.sort("kickoff_utc")
    right = obs.sort("obs_utc")
    return left.join_asof(
        right,
        left_on="kickoff_utc",
        right_on="obs_utc",
        strategy="nearest",
        tolerance=timedelta(minutes=90),
    )


def _rename_measures(df: pl.DataFrame) -> pl.DataFrame:
    mapping = {
        "temperature_2m": "temperature_c",
        "relative_humidity_2m": "relative_humidity",
        "dew_point_2m": "dew_point_c",
        "apparent_temperature": "apparent_temperature_c",
        "precipitation": "precipitation_mm",
        "snowfall": "snowfall_cm",
        "surface_pressure": "surface_pressure_hpa",
        "wind_speed_10m": "wind_speed_kmh",
        "wind_gusts_10m": "wind_gust_kmh",
        "wind_direction_10m": "wind_direction_deg",
        "cloud_cover": "cloud_cover_pct",
    }
    present = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(present)


def build(season: int, **_: Any) -> pl.DataFrame:
    """Per-game weather for one season. Requires schedules + stadiums locally."""
    from ..storage import read_dataset
    from .stadiums import geography_for_games

    cfg = get_settings()
    mf = get_manifest()

    games = read_dataset(mf["schedules"], root=RAW_DIR, seasons=[season])
    stadium_dim = read_dataset(mf["stadiums"], root=RAW_DIR)

    keep = [c for c in ("game_id", "season", "week", "gameday", "kickoff_utc",
                        "stadium_id", "roof", "home_team", "away_team") if c in games.columns]
    games = games.select(keep).drop_nulls("kickoff_utc")
    if games.is_empty():
        log.warning("season %s has no games with a resolvable kickoff time", season)
        return pl.DataFrame()

    games = geography_for_games(games, stadium_dim)

    # Roof is per game, from schedules. Null roof is treated as outdoor but
    # flagged, because guessing indoor would suppress real weather signal.
    games = games.with_columns(
        pl.col("roof").fill_null("unknown").alias("roof_resolved"),
    ).with_columns(
        pl.col("roof_resolved").is_in(list(INDOOR_ROOFS)).alias("is_indoor"),
        (pl.col("roof_resolved") == "unknown").alias("roof_unknown"),
    )

    outdoor = games.filter(~pl.col("is_indoor") & pl.col("latitude").is_not_null())
    indoor = games.filter(pl.col("is_indoor"))
    ungeocoded = games.filter(~pl.col("is_indoor") & pl.col("latitude").is_null())

    if ungeocoded.height:
        log.warning(
            "season %s: %d outdoor games have no venue coordinates; their weather stays null",
            season, ungeocoded.height,
        )

    observed_parts: list[pl.DataFrame] = []
    if cfg.weather_provider == "none":
        log.info("weather provider disabled; emitting flags only")
    elif outdoor.height:
        by_venue: dict[tuple[float, float], list[Any]] = defaultdict(list)
        for row in outdoor.select("latitude", "longitude").unique().iter_rows():
            by_venue[(round(row[0], 4), round(row[1], 4))] = []

        for (lat, lon) in by_venue:
            venue_games = outdoor.filter(
                (pl.col("latitude").round(4) == lat) & (pl.col("longitude").round(4) == lon)
            )
            start = venue_games.get_column("gameday").cast(pl.Date).min()
            end = venue_games.get_column("gameday").cast(pl.Date).max()
            try:
                obs = _fetch_venue_season(
                    lat, lon,
                    (start - timedelta(days=1)).isoformat(),
                    (end + timedelta(days=1)).isoformat(),
                )
            except Exception as exc:  # network/provider failure must not corrupt the lake
                log.error("weather fetch failed for (%s, %s) season %s: %s", lat, lon, season, exc)
                obs = pl.DataFrame()
            observed_parts.append(_nearest_hour(venue_games, obs))
    else:
        log.info("season %s: no outdoor games needing a weather fetch", season)

    frames: list[pl.DataFrame] = []
    if observed_parts:
        frames.append(_rename_measures(pl.concat(observed_parts, how="diagonal_relaxed")))
    if indoor.height:
        frames.append(
            indoor.with_columns(
                [pl.lit(v).alias(k) for k, v in INDOOR_DEFAULTS.items()]
            ).with_columns(pl.lit(None, dtype=pl.Float64).alias("surface_pressure_hpa"))
        )
    if ungeocoded.height:
        frames.append(ungeocoded)

    if not frames:
        return pl.DataFrame()

    out = pl.concat(frames, how="diagonal_relaxed")

    out = out.with_columns(
        pl.lit(cfg.weather_provider).alias("weather_provider"),
        pl.lit(True).alias("weather_is_reanalysis"),
        pl.when(pl.col("is_indoor"))
        .then(pl.lit("indoor_constant"))
        .when(pl.col("temperature_c").is_not_null() if "temperature_c" in out.columns else pl.lit(False))
        .then(pl.lit("era5_nearest_hour"))
        .otherwise(pl.lit("unavailable"))
        .alias("weather_method"),
    )

    return _cross_check_against_schedules(out, season)


def _cross_check_against_schedules(weather: pl.DataFrame, season: int) -> pl.DataFrame:
    """Compare ERA5 against the temp/wind nflverse ships, and record the gap.

    nflverse ``temp``/``wind`` come from the official gamebook for some games.
    Per the manifest's source priority, official NFL data outranks a third-party
    reanalysis -- so where the gamebook has a value we surface it and mark the
    disagreement rather than quietly preferring our own fetch.
    """
    from ..storage import read_dataset

    mf = get_manifest()
    sched = read_dataset(mf["schedules"], root=RAW_DIR, seasons=[season])
    cols = [c for c in ("game_id", "temp", "wind") if c in sched.columns]
    if len(cols) < 3 or "temperature_c" not in weather.columns:
        return weather

    official = sched.select(cols).rename({"temp": "gamebook_temp_f", "wind": "gamebook_wind_mph"})
    out = weather.join(official, on="game_id", how="left")

    return out.with_columns(
        ((pl.col("gamebook_temp_f") - 32) * 5 / 9).alias("gamebook_temp_c"),
    ).with_columns(
        (pl.col("gamebook_temp_c") - pl.col("temperature_c")).abs().alias("temp_disagreement_c"),
        pl.when(pl.col("gamebook_temp_f").is_not_null())
        .then(pl.lit("official_nfl"))
        .otherwise(pl.lit("reputable_public"))
        .alias("weather_winning_tier"),
    )


def summarize_coverage(weather: pl.DataFrame) -> dict[str, Any]:
    """Fill rates, for the quality report."""
    if weather.is_empty():
        return {"games": 0}
    total = weather.height
    have_temp = (
        weather.get_column("temperature_c").is_not_null().sum()
        if "temperature_c" in weather.columns else 0
    )
    return {
        "games": total,
        "indoor": int(weather.get_column("is_indoor").sum()) if "is_indoor" in weather.columns else 0,
        "with_temperature": int(have_temp),
        "temperature_fill_rate": round(float(have_temp) / total, 4),
    }


def iter_missing_coordinates(stadium_dim: pl.DataFrame) -> Iterable[str]:
    if "has_coordinates" not in stadium_dim.columns:
        return []
    return (
        stadium_dim.filter(~pl.col("has_coordinates"))
        .get_column("stadium_id")
        .to_list()
    )
