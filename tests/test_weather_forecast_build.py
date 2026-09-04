"""Season-builder tests for live NFL weather forecast snapshots."""

from datetime import datetime, timezone

import polars as pl

from nfldl import storage
from nfldl.sources import stadiums, weather_forecast


def _utc(
    day: int,
    hour: int = 18,
) -> datetime:
    return datetime(
        2026,
        9,
        day,
        hour,
        tzinfo=timezone.utc,
    )


def _schedules() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [
                "past",
                "outdoor_1",
                "outdoor_2",
                "indoor",
                "too_far",
            ],
            "season": [2026] * 5,
            "week": [1, 1, 1, 1, 3],
            "gameday": [
                "2026-09-02",
                "2026-09-09",
                "2026-09-10",
                "2026-09-11",
                "2026-09-25",
            ],
            "kickoff_utc": [
                _utc(2),
                _utc(9),
                _utc(10),
                _utc(11),
                _utc(25),
            ],
            "stadium_id": [
                "past_stadium",
                "sea",
                "sea",
                "dome",
                "far_stadium",
            ],
            "roof": [
                "outdoors",
                "outdoors",
                "outdoors",
                "dome",
                "outdoors",
            ],
            "home_team": [
                "AAA",
                "SEA",
                "SEA",
                "DET",
                "BBB",
            ],
            "away_team": [
                "BBB",
                "NE",
                "LAR",
                "CHI",
                "CCC",
            ],
        }
    )


def _stadium_dimension() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stadium_id": [
                "past_stadium",
                "sea",
                "dome",
                "far_stadium",
            ],
        }
    )


def _geography_for_games(
    games: pl.DataFrame,
    _stadium_dim: pl.DataFrame,
) -> pl.DataFrame:
    coords = pl.DataFrame(
        {
            "stadium_id": [
                "past_stadium",
                "sea",
                "dome",
                "far_stadium",
            ],
            "latitude": [
                40.0,
                47.5952,
                42.34,
                35.0,
            ],
            "longitude": [
                -75.0,
                -122.3316,
                -83.0456,
                -90.0,
            ],
        }
    )

    return games.join(
        coords,
        on="stadium_id",
        how="left",
    )


def _forecast_frame() -> pl.DataFrame:
    times = [
        _utc(9),
        _utc(10),
    ]

    return pl.DataFrame(
        {
            "forecast_valid_at_utc": times,
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
                15.0,
            ],
            "precipitation": [
                0.0,
                0.0,
            ],
            "snowfall": [
                0.0,
                0.0,
            ],
            "surface_pressure": [
                1012.0,
                1011.0,
            ],
            "wind_speed_10m": [
                8.0,
                9.0,
            ],
            "wind_gusts_10m": [
                14.0,
                15.0,
            ],
            "wind_direction_10m": [
                220.0,
                225.0,
            ],
            "cloud_cover": [
                35.0,
                40.0,
            ],
            "visibility": [
                24000.0,
                23000.0,
            ],
        }
    )


def _patch_local_inputs(
    monkeypatch,
):
    schedules = _schedules()
    stadium_dim = _stadium_dimension()

    def fake_read_dataset(
        spec,
        **_,
    ):
        if spec.name == "schedules":
            return schedules
        if spec.name == "stadiums":
            return stadium_dim
        raise AssertionError(
            f"unexpected dataset read: {spec.name}"
        )

    monkeypatch.setattr(
        storage,
        "read_dataset",
        fake_read_dataset,
    )

    monkeypatch.setattr(
        stadiums,
        "geography_for_games",
        _geography_for_games,
    )


def test_build_only_snapshots_upcoming_games_inside_live_horizon(
    monkeypatch,
):
    _patch_local_inputs(
        monkeypatch,
    )

    retrieved = _utc(
        3,
        18,
    )

    monkeypatch.setattr(
        weather_forecast,
        "_utc_now",
        lambda: retrieved,
        raising=False,
    )

    calls = []

    def fake_fetch(
        lat,
        lon,
    ):
        calls.append(
            (
                round(lat, 4),
                round(lon, 4),
            )
        )
        return _forecast_frame()

    monkeypatch.setattr(
        weather_forecast,
        "_fetch_venue_forecast",
        fake_fetch,
    )

    out = weather_forecast.build(
        2026
    ).sort("game_id")

    assert set(
        out.get_column(
            "game_id"
        ).to_list()
    ) == {
        "outdoor_1",
        "outdoor_2",
        "indoor",
    }

    # Both outdoor games share one venue and therefore one API request.
    assert calls == [
        (
            47.5952,
            -122.3316,
        )
    ]

    assert out.get_column(
        "retrieved_at_utc"
    ).n_unique() == 1

    assert out.get_column(
        "retrieved_at_utc"
    )[0] == retrieved

    assert out.filter(
        pl.col("game_id")
        == "outdoor_1"
    ).get_column(
        "temperature_c"
    )[0] == 18.0


def test_build_indoor_game_uses_neutral_conditions_without_api_call(
    monkeypatch,
):
    _patch_local_inputs(
        monkeypatch,
    )

    retrieved = _utc(
        3,
        18,
    )

    monkeypatch.setattr(
        weather_forecast,
        "_utc_now",
        lambda: retrieved,
        raising=False,
    )

    monkeypatch.setattr(
        weather_forecast,
        "_fetch_venue_forecast",
        lambda *_args, **_kwargs: _forecast_frame(),
    )

    out = weather_forecast.build(
        2026
    )

    indoor = out.filter(
        pl.col("game_id")
        == "indoor"
    )

    assert indoor.height == 1
    assert indoor.get_column(
        "is_indoor"
    )[0] is True
    assert indoor.get_column(
        "temperature_c"
    )[0] == 21.0
    assert indoor.get_column(
        "precipitation_mm"
    )[0] == 0.0
    assert indoor.get_column(
        "wind_speed_kmh"
    )[0] == 0.0
    assert indoor.get_column(
        "weather_method"
    )[0] == "indoor_constant"


def test_build_never_emits_nonpositive_forecast_lead_time(
    monkeypatch,
):
    _patch_local_inputs(
        monkeypatch,
    )

    retrieved = _utc(
        3,
        18,
    )

    monkeypatch.setattr(
        weather_forecast,
        "_utc_now",
        lambda: retrieved,
        raising=False,
    )

    monkeypatch.setattr(
        weather_forecast,
        "_fetch_venue_forecast",
        lambda *_args, **_kwargs: _forecast_frame(),
    )

    out = weather_forecast.build(
        2026
    )

    assert out.filter(
        pl.col("forecast_lead_hours")
        <= 0
    ).is_empty()

    assert out.filter(
        pl.col("kickoff_utc")
        > (
            pl.lit(retrieved)
            + pl.duration(
                days=16,
            )
        )
    ).is_empty()


def test_build_marks_missing_provider_forecast_as_unavailable(
    monkeypatch,
):
    _patch_local_inputs(
        monkeypatch,
    )

    retrieved = _utc(
        3,
        18,
    )

    monkeypatch.setattr(
        weather_forecast,
        "_utc_now",
        lambda: retrieved,
        raising=False,
    )

    monkeypatch.setattr(
        weather_forecast,
        "_fetch_venue_forecast",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )

    out = weather_forecast.build(
        2026
    )

    outdoor = out.filter(
        ~pl.col("is_indoor")
    )

    assert outdoor.height == 2

    assert set(
        outdoor.get_column(
            "weather_method"
        ).to_list()
    ) == {
        "unavailable"
    }

    assert outdoor.get_column(
        "temperature_c"
    ).null_count() == outdoor.height

    assert (
        outdoor.get_column(
            "forecast_valid_at_utc"
        ).null_count()
        == outdoor.height
    )
