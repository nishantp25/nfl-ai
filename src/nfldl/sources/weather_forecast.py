"""Point-in-time pregame weather forecast snapshots.

This module is intentionally separate from :mod:`nfldl.sources.weather`.

``weather`` contains realized ERA5 reanalysis conditions.
``weather_forecast`` contains forecasts actually retrieved before games.

For live forecasts, ``retrieved_at_utc`` is the knowledge timestamp. That
timestamp is part of the dataset primary key, allowing multiple immutable
forecast snapshots for the same NFL game.

Historical forecast reconstruction is intentionally not implemented here yet.
Provider archives have model-run timing semantics that must be verified before
they can be claimed as point-in-time safe.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import polars as pl

from ..config import RAW_DIR
from ..http import get_json
from ..manifest import get_manifest

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "snowfall",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "cloud_cover",
    "visibility",
]


INDOOR_ROOFS = {
    "dome",
    "closed",
}

INDOOR_DEFAULTS = {
    "temperature_c": 21.0,
    "relative_humidity": 50.0,
    "apparent_temperature_c": 21.0,
    "precipitation_probability_pct": 0.0,
    "precipitation_mm": 0.0,
    "snowfall_cm": 0.0,
    "surface_pressure_hpa": None,
    "wind_speed_kmh": 0.0,
    "wind_gust_kmh": 0.0,
    "wind_direction_deg": None,
    "cloud_cover_pct": 0.0,
    "visibility_m": None,
}


def _utc_now() -> datetime:
    """Current UTC instant, isolated so snapshot timing is testable."""
    return datetime.now(timezone.utc)



def _fetch_venue_forecast(
    lat: float,
    lon: float,
) -> pl.DataFrame:
    """Fetch the current Open-Meteo hourly forecast for one venue.

    The public Forecast API can return up to 16 days. Timestamps are requested
    in UTC so they can be compared directly with schedule ``kickoff_utc``.
    """
    payload = get_json(
        FORECAST_URL,
        params={
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "hourly": ",".join(HOURLY_VARS),
            "forecast_days": 16,
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        },
    )

    hourly = payload.get("hourly") or {}

    if not hourly.get("time"):
        return pl.DataFrame()

    return (
        pl.DataFrame(hourly)
        .with_columns(
            pl.col("time")
            .str.strptime(
                pl.Datetime("us"),
                "%Y-%m-%dT%H:%M",
                strict=False,
            )
            .dt.replace_time_zone("UTC")
            .alias("forecast_valid_at_utc")
        )
        .drop("time")
        .with_columns(
            pl.lit(round(lat, 4)).alias("_lat"),
            pl.lit(round(lon, 4)).alias("_lon"),
        )
    )


def _nearest_kickoff_forecast(
    games: pl.DataFrame,
    forecast: pl.DataFrame,
) -> pl.DataFrame:
    """Attach the forecast hour nearest each game's kickoff.

    Hourly Open-Meteo output is matched within 90 minutes, mirroring the
    realized-weather layer while keeping forecast-valid time explicit.
    """
    if forecast.is_empty():
        return games.with_columns(
            [
                pl.lit(
                    None,
                    dtype=pl.Float64,
                ).alias(variable)
                for variable in HOURLY_VARS
            ]
            + [
                pl.lit(
                    None,
                    dtype=pl.Datetime(
                        "us",
                        "UTC",
                    ),
                ).alias(
                    "forecast_valid_at_utc"
                )
            ]
        )

    left = games.sort("kickoff_utc")
    right = forecast.sort("forecast_valid_at_utc")

    return left.join_asof(
        right,
        left_on="kickoff_utc",
        right_on="forecast_valid_at_utc",
        strategy="nearest",
        tolerance=timedelta(
            minutes=90,
        ),
    )


def _rename_forecast_measures(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Rename provider fields to stable, unit-explicit model columns."""
    mapping = {
        "temperature_2m": "temperature_c",
        "relative_humidity_2m": "relative_humidity",
        "apparent_temperature": "apparent_temperature_c",
        "precipitation_probability": "precipitation_probability_pct",
        "precipitation": "precipitation_mm",
        "snowfall": "snowfall_cm",
        "surface_pressure": "surface_pressure_hpa",
        "wind_speed_10m": "wind_speed_kmh",
        "wind_gusts_10m": "wind_gust_kmh",
        "wind_direction_10m": "wind_direction_deg",
        "cloud_cover": "cloud_cover_pct",
        "visibility": "visibility_m",
    }

    present = {
        source: target
        for source, target in mapping.items()
        if source in df.columns
    }

    return df.rename(
        present
    )


def _stamp_snapshot_metadata(
    df: pl.DataFrame,
    *,
    retrieved_at_utc: datetime,
) -> pl.DataFrame:
    """Attach provenance and point-in-time metadata to one live snapshot."""
    if retrieved_at_utc.tzinfo is None:
        raise ValueError(
            "retrieved_at_utc must be timezone-aware"
        )

    retrieved = retrieved_at_utc.astimezone(
        timezone.utc
    )

    retrieved_expr = (
        pl.lit(retrieved)
        .cast(
            pl.Datetime(
                "us",
                "UTC",
            )
        )
    )

    out = df.with_columns(
        retrieved_expr.alias(
            "retrieved_at_utc"
        ),
        pl.lit(
            "open_meteo_forecast"
        ).alias(
            "weather_provider"
        ),
        pl.lit(
            False
        ).alias(
            "weather_is_reanalysis"
        ),
        pl.lit(
            "open_meteo_live_forecast"
        ).alias(
            "weather_method"
        ),
    )

    if "kickoff_utc" in out.columns:
        out = out.with_columns(
            (
                (
                    pl.col("kickoff_utc")
                    - retrieved_expr
                )
                .dt.total_minutes()
                .cast(pl.Float64)
                / 60.0
            ).alias(
                "forecast_lead_hours"
            )
        )

    return out


def build(
    season: int,
    **_: Any,
) -> pl.DataFrame:
    """Build one immutable live forecast snapshot for upcoming NFL games.

    Only games strictly after snapshot start and no more than 16 days away are
    considered. Outdoor games sharing coordinates reuse one Open-Meteo request.
    Indoor games receive explicit neutral constants and never trigger a weather
    API request.

    ``retrieved_at_utc`` is captured after all provider calls complete. That is
    deliberately conservative: every row in the returned snapshot was certainly
    knowable by that instant.
    """
    from .. import storage
    from . import stadiums

    mf = get_manifest()

    schedules = storage.read_dataset(
        mf["schedules"],
        root=RAW_DIR,
        seasons=[season],
    )

    stadium_dim = storage.read_dataset(
        mf["stadiums"],
        root=RAW_DIR,
    )

    keep = [
        column
        for column in (
            "game_id",
            "season",
            "week",
            "gameday",
            "kickoff_utc",
            "stadium_id",
            "roof",
            "home_team",
            "away_team",
        )
        if column in schedules.columns
    ]

    games = (
        schedules
        .select(keep)
        .drop_nulls("kickoff_utc")
    )

    if games.is_empty():
        return pl.DataFrame()

    snapshot_started_at = _utc_now()
    horizon_end = (
        snapshot_started_at
        + timedelta(days=16)
    )

    # A live snapshot is useful only for genuinely future games inside the
    # provider's supported forecast horizon.
    games = games.filter(
        (pl.col("kickoff_utc") > pl.lit(snapshot_started_at))
        & (pl.col("kickoff_utc") <= pl.lit(horizon_end))
    )

    if games.is_empty():
        return pl.DataFrame()

    games = stadiums.geography_for_games(
        games,
        stadium_dim,
    )

    if "roof" in games.columns:
        games = games.with_columns(
            pl.col("roof")
            .fill_null("unknown")
            .cast(pl.String)
            .str.to_lowercase()
            .alias("roof_resolved")
        )
    else:
        games = games.with_columns(
            pl.lit("unknown").alias(
                "roof_resolved"
            )
        )

    games = games.with_columns(
        pl.col("roof_resolved")
        .is_in(list(INDOOR_ROOFS))
        .alias("is_indoor"),
        (
            pl.col("roof_resolved")
            == "unknown"
        ).alias("roof_unknown"),
    )

    indoor = games.filter(
        pl.col("is_indoor")
    )

    outdoor = games.filter(
        ~pl.col("is_indoor")
        & pl.col("latitude").is_not_null()
        & pl.col("longitude").is_not_null()
    )

    ungeocoded = games.filter(
        ~pl.col("is_indoor")
        & (
            pl.col("latitude").is_null()
            | pl.col("longitude").is_null()
        )
    )

    parts: list[pl.DataFrame] = []

    # One live provider request per unique coordinate pair.
    if outdoor.height:
        venues = (
            outdoor
            .select(
                pl.col("latitude").round(4),
                pl.col("longitude").round(4),
            )
            .unique(
                maintain_order=True
            )
        )

        for lat, lon in venues.iter_rows():
            venue_games = outdoor.filter(
                (pl.col("latitude").round(4) == lat)
                & (pl.col("longitude").round(4) == lon)
            )

            forecast = _fetch_venue_forecast(
                float(lat),
                float(lon),
            )

            matched = _nearest_kickoff_forecast(
                venue_games,
                forecast,
            )

            parts.append(
                _rename_forecast_measures(
                    matched
                )
            )

    if indoor.height:
        indoor_part = indoor.with_columns(
            [
                pl.lit(value).alias(column)
                for column, value in INDOOR_DEFAULTS.items()
            ]
        ).with_columns(
            pl.lit(
                None,
                dtype=pl.Datetime(
                    "us",
                    "UTC",
                ),
            ).alias(
                "forecast_valid_at_utc"
            )
        )

        parts.append(
            indoor_part
        )

    if ungeocoded.height:
        unavailable = ungeocoded.with_columns(
            [
                pl.lit(
                    None,
                    dtype=pl.Float64,
                ).alias(column)
                for column in INDOOR_DEFAULTS
            ]
        ).with_columns(
            pl.lit(
                None,
                dtype=pl.Datetime(
                    "us",
                    "UTC",
                ),
            ).alias(
                "forecast_valid_at_utc"
            )
        )

        parts.append(
            unavailable
        )

    if not parts:
        return pl.DataFrame()

    snapshot = pl.concat(
        parts,
        how="diagonal_relaxed",
    )

    # Capture knowledge time only after all network calls have completed.
    retrieved_at_utc = _utc_now()

    snapshot = _stamp_snapshot_metadata(
        snapshot,
        retrieved_at_utc=retrieved_at_utc,
    )

    # Override the generic live-provider method for rows that did not come
    # from an outdoor forecast response.
    snapshot = snapshot.with_columns(
        pl.when(
            pl.col("is_indoor")
        )
        .then(
            pl.lit("indoor_constant")
        )
        .when(
            pl.col("latitude").is_null()
            | pl.col("longitude").is_null()
            | pl.col("forecast_valid_at_utc").is_null()
        )
        .then(
            pl.lit("unavailable")
        )
        .otherwise(
            pl.col("weather_method")
        )
        .alias(
            "weather_method"
        )
    )

    # Final hard guard: no row collected by this live builder may represent
    # information at or after the game's own kickoff.
    snapshot = snapshot.filter(
        pl.col("forecast_lead_hours") > 0
    )

    return snapshot
