"""Contract tests for point-in-time pregame weather forecasts.

Historical realized weather and pregame forecast weather are intentionally
separate datasets. A forecast snapshot is information known at retrieval time,
so multiple snapshots may legitimately exist for the same NFL game.
"""

from nfldl.manifest import load_manifest
from nfldl.registry import REGISTRY, resolve_order


def test_weather_forecast_is_a_separate_manifest_dataset():
    manifest = load_manifest()

    assert "weather_forecast" in manifest.names()

    spec = manifest["weather_forecast"]

    assert not spec.required
    assert spec.primary_key == (
        "game_id",
        "retrieved_at_utc",
    )
    assert spec.timestamp_field == "retrieved_at_utc"

    # Documentation of provider availability is not schema/coverage verification.
    assert spec.verified.start is None
    assert spec.verified.end is None


def test_weather_forecast_adapter_requires_game_and_venue_context():
    assert "weather_forecast" in REGISTRY

    adapter = REGISTRY["weather_forecast"]

    assert adapter.needs_local == (
        "schedules",
        "stadiums",
    )


def test_weather_forecast_dependency_order_is_point_in_time_ready():
    order = resolve_order(
        [
            "weather_forecast",
        ]
    )

    assert order.index("schedules") < order.index("weather_forecast")
    assert order.index("stadiums") < order.index("weather_forecast")
