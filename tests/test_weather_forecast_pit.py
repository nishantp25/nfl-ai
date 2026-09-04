"""Point-in-time tests for pregame weather forecast snapshots."""

from datetime import datetime, timezone

import polars as pl

from nfldl import pit


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


def test_weather_forecast_is_classified_as_pregame():
    assert pit.DATASET_AVAILABILITY["weather_forecast"] == "pregame"


def test_weather_forecast_known_at_preserves_actual_retrieval_time():
    kickoff = _utc(2026, 9, 10, 18)
    retrieved_after_kickoff = _utc(2026, 9, 10, 19)

    df = pl.DataFrame(
        {
            "game_id": ["g1"],
            "kickoff_utc": [kickoff],
            "retrieved_at_utc": [retrieved_after_kickoff],
        }
    )

    out = pit.knowledge_time(
        df,
        "weather_forecast",
        report_col="retrieved_at_utc",
    )

    assert out.get_column("_known_at")[0] == retrieved_after_kickoff
    assert out.get_column("_availability")[0] == "pregame"


def test_asof_join_selects_latest_strictly_pregame_forecast():
    kickoff = _utc(2026, 9, 10, 18)

    games = pl.DataFrame(
        {
            "game_id": ["g1"],
            "kickoff_utc": [kickoff],
        }
    )

    forecasts = pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g1", "g1"],
            "kickoff_utc": [kickoff] * 4,
            "retrieved_at_utc": [
                _utc(2026, 9, 10, 12),
                _utc(2026, 9, 10, 17),
                _utc(2026, 9, 10, 18),
                _utc(2026, 9, 10, 19),
            ],
            "temperature_c": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
        }
    )

    forecasts = pit.knowledge_time(
        forecasts,
        "weather_forecast",
        report_col="retrieved_at_utc",
    )

    joined = pit.asof_join(
        games,
        forecasts,
        by=["game_id"],
    )

    pit.assert_no_future_rows(joined)

    assert joined.get_column("temperature_c")[0] == 20.0
    assert joined.get_column("_known_at")[0] == _utc(
        2026,
        9,
        10,
        17,
    )
