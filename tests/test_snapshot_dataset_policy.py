"""Policy tests for snapshot-style prospective datasets."""

from nfldl import ingest
from nfldl.manifest import get_manifest


def test_weather_forecast_declares_snapshot_storage_policy():
    spec = get_manifest()["weather_forecast"]

    assert spec.raw["storage_mode"] == "append"
    assert spec.raw["season_horizon"] == "league_year"


def test_weather_forecast_can_collect_current_league_year_before_opener(
    monkeypatch,
):
    spec = get_manifest()["weather_forecast"]

    # Sep. 2, 2026-style situation:
    # the 2026 playing season has not opened yet, but league year 2026 exists.
    monkeypatch.setattr(
        ingest,
        "nfl_season_of",
        lambda: 2025,
    )
    monkeypatch.setattr(
        ingest,
        "league_year_of",
        lambda: 2026,
        raising=False,
    )

    available, unavailable = ingest._seasons_for(
        spec,
        [2026],
    )

    assert available == [2026]
    assert unavailable == []
