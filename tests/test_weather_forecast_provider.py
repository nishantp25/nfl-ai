"""Behavior tests for live Open-Meteo NFL weather forecast snapshots."""

from datetime import datetime, timezone

import polars as pl

from nfldl.sources import weather_forecast


def _utc(
    year: int,
    month: int,
    day: int,
    hour: int,
) -> datetime:
    return datetime(
        year,
        month,
        day,
        hour,
        tzinfo=timezone.utc,
    )


def test_live_forecast_request_uses_utc_and_full_supported_horizon(
    monkeypatch,
):
    captured = {}

    def fake_get_json(url, *, params):
        captured["url"] = url
        captured["params"] = params

        return {
            "hourly": {
                "time": [
                    "2026-09-09T23:00",
                    "2026-09-10T00:00",
                ],
                "temperature_2m": [
                    18.0,
                    17.0,
                ],
                "relative_humidity_2m": [
                    60.0,
                    63.0,
                ],
                "apparent_temperature": [
                    18.0,
                    17.0,
                ],
                "precipitation_probability": [
                    10.0,
                    20.0,
                ],
                "precipitation": [
                    0.0,
                    0.1,
                ],
                "snowfall": [
                    0.0,
                    0.0,
                ],
                "surface_pressure": [
                    1012.0,
                    1012.5,
                ],
                "wind_speed_10m": [
                    8.0,
                    9.0,
                ],
                "wind_gusts_10m": [
                    14.0,
                    16.0,
                ],
                "wind_direction_10m": [
                    220.0,
                    225.0,
                ],
                "cloud_cover": [
                    35.0,
                    42.0,
                ],
                "visibility": [
                    24000.0,
                    23000.0,
                ],
            }
        }

    monkeypatch.setattr(
        weather_forecast,
        "get_json",
        fake_get_json,
        raising=False,
    )

    out = weather_forecast._fetch_venue_forecast(
        47.5952,
        -122.3316,
    )

    params = captured["params"]

    assert captured["url"] == (
        "https://api.open-meteo.com/v1/forecast"
    )
    assert params["timezone"] == "UTC"
    assert params["forecast_days"] == 16
    assert params["temperature_unit"] == "celsius"
    assert params["wind_speed_unit"] == "kmh"
    assert params["precipitation_unit"] == "mm"

    requested = set(
        params["hourly"].split(",")
    )

    assert {
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
    } <= requested

    assert out.height == 2
    assert "forecast_valid_at_utc" in out.columns
    assert out.schema["forecast_valid_at_utc"] == pl.Datetime(
        "us",
        "UTC",
    )


def test_kickoff_join_uses_nearest_forecast_hour():
    kickoff = _utc(
        2026,
        9,
        9,
        23,
    )

    games = pl.DataFrame(
        {
            "game_id": ["g1"],
            "kickoff_utc": [
                kickoff.replace(
                    minute=20,
                )
            ],
        }
    )

    forecast = pl.DataFrame(
        {
            "forecast_valid_at_utc": [
                _utc(
                    2026,
                    9,
                    9,
                    23,
                ),
                _utc(
                    2026,
                    9,
                    10,
                    0,
                ),
            ],
            "temperature_2m": [
                18.0,
                14.0,
            ],
        }
    )

    out = weather_forecast._nearest_kickoff_forecast(
        games,
        forecast,
    )

    assert out.get_column(
        "temperature_2m"
    )[0] == 18.0

    assert out.get_column(
        "forecast_valid_at_utc"
    )[0] == _utc(
        2026,
        9,
        9,
        23,
    )


def test_measure_names_are_model_ready():
    df = pl.DataFrame(
        {
            "temperature_2m": [18.0],
            "relative_humidity_2m": [60.0],
            "apparent_temperature": [18.0],
            "precipitation_probability": [25.0],
            "precipitation": [0.3],
            "snowfall": [0.0],
            "surface_pressure": [1012.0],
            "wind_speed_10m": [8.0],
            "wind_gusts_10m": [14.0],
            "wind_direction_10m": [220.0],
            "cloud_cover": [35.0],
            "visibility": [24000.0],
        }
    )

    out = weather_forecast._rename_forecast_measures(
        df
    )

    expected = {
        "temperature_c",
        "relative_humidity",
        "apparent_temperature_c",
        "precipitation_probability_pct",
        "precipitation_mm",
        "snowfall_cm",
        "surface_pressure_hpa",
        "wind_speed_kmh",
        "wind_gust_kmh",
        "wind_direction_deg",
        "cloud_cover_pct",
        "visibility_m",
    }

    assert expected <= set(out.columns)


def test_snapshot_metadata_uses_retrieval_as_knowledge_time():
    retrieved = _utc(
        2026,
        9,
        3,
        18,
    )

    kickoff = _utc(
        2026,
        9,
        9,
        23,
    )

    df = pl.DataFrame(
        {
            "game_id": ["g1"],
            "kickoff_utc": [kickoff],
            "temperature_c": [18.0],
        }
    )

    out = weather_forecast._stamp_snapshot_metadata(
        df,
        retrieved_at_utc=retrieved,
    )

    assert out.get_column(
        "retrieved_at_utc"
    )[0] == retrieved

    assert out.get_column(
        "weather_is_reanalysis"
    )[0] is False

    assert out.get_column(
        "weather_method"
    )[0] == "open_meteo_live_forecast"

    assert out.get_column(
        "forecast_lead_hours"
    )[0] == 149.0
