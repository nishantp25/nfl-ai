"""Regression tests for legacy nflverse depth-chart normalization.

Live nflverse inspection established that the 2001-2024 depth-chart schema has
two important quirks:

1. The upstream seasonal files contain exact duplicate rows.
2. ``week`` is legitimately null for ``game_type == "SBBYE"`` rows.

After exact-row deduplication, the legacy identity that is unique across every
season from 2001 through 2024 is:

    season
    week
    club_code
    gsis_id
    depth_position
    game_type
    depth_team
    formation

A player may legitimately appear in both Offense and Special Teams in the same
week, so ``formation`` is part of the identity rather than optional metadata.
"""

from __future__ import annotations

import polars as pl

from nfldl import quality
from nfldl.manifest import load_manifest
from nfldl.sources import nflverse

LEGACY_KEY = (
    "season",
    "week",
    "club_code",
    "gsis_id",
    "depth_position",
    "game_type",
    "depth_team",
    "formation",
)


def _legacy_depth_frame(
    *,
    week: int | None,
    game_type: str,
    gsis_id: str | None = "00-0000001",
) -> pl.DataFrame:
    """Return one row satisfying the legacy depth-chart schema."""
    return pl.DataFrame(
        {
            "season": [2024],
            "week": [week],
            "club_code": ["KC"],
            "gsis_id": [gsis_id],
            "depth_position": ["RB"],
            "game_type": [game_type],
            "depth_team": ["2"],
            "formation": ["Offense"],
        }
    )


def test_legacy_depth_chart_contract_matches_live_identity():
    """The committed legacy key must reflect the full upstream identity."""
    spec = load_manifest()["depth_charts"]
    contract = spec.schema_for_season(2024)

    assert contract.primary_key == LEGACY_KEY
    assert contract.timestamp_field is None

    assert len(contract.nullable_key_rules) == 1

    rule = contract.nullable_key_rules[0]

    assert rule.column == "week"
    assert dict(rule.when) == {
        "game_type": "SBBYE",
    }


def test_2025_depth_chart_contract_has_no_legacy_null_exception():
    """The timestamped 2025+ feed must not inherit the SBBYE week rule."""
    spec = load_manifest()["depth_charts"]
    contract = spec.schema_for_season(2025)

    assert contract.nullable_key_rules == ()


def test_fetch_depth_charts_removes_only_exact_duplicates(monkeypatch):
    """Canonicalization removes repeated source rows but preserves real records."""
    raw = pl.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [1, 1, 1],
            "club_code": ["CHI", "CHI", "CHI"],
            "game_type": ["REG", "REG", "REG"],
            "depth_team": ["3", "3", "3"],
            "last_name": ["Homer", "Homer", "Homer"],
            "first_name": ["Travis", "Travis", "Travis"],
            "football_name": ["Travis", "Travis", "Travis"],
            "formation": [
                "Offense",
                "Offense",
                "Special Teams",
            ],
            "gsis_id": [
                "00-0035594",
                "00-0035594",
                "00-0035594",
            ],
            "jersey_number": ["20", "20", "20"],
            "position": ["RB", "RB", "RB"],
            "elias_id": ["HOM655558", "HOM655558", "HOM655558"],
            "depth_position": ["", "", ""],
            "full_name": [
                "Travis Homer",
                "Travis Homer",
                "Travis Homer",
            ],
        }
    )

    monkeypatch.setattr(
        nflverse,
        "_with_fallback",
        lambda *_args, **_kwargs: raw,
    )

    result = nflverse.fetch_depth_charts(2023)

    assert result.height == 2
    assert result.unique().height == 2

    assert set(
        result.get_column("formation").to_list()
    ) == {
        "Offense",
        "Special Teams",
    }


def test_sbbye_may_have_null_week_in_legacy_depth_chart():
    """A null week is structural, not corrupt, on legacy SBBYE rows."""
    spec = load_manifest()["depth_charts"]

    df = _legacy_depth_frame(
        week=None,
        game_type="SBBYE",
    )

    result = quality.check_null_rates(
        df,
        spec,
    )

    assert result.passed
    assert result.severity != "error"


def test_regular_game_may_not_have_null_week():
    """The SBBYE exception must never turn ordinary null weeks into valid data."""
    spec = load_manifest()["depth_charts"]

    df = _legacy_depth_frame(
        week=None,
        game_type="REG",
    )

    result = quality.check_null_rates(
        df,
        spec,
    )

    assert not result.passed
    assert result.severity == "error"

    assert "2024:week" in result.data["key_nulls"]


def test_sbbye_exception_does_not_allow_other_null_key_fields():
    """Conditional nullability applies only to the explicitly declared column."""
    spec = load_manifest()["depth_charts"]

    df = _legacy_depth_frame(
        week=None,
        game_type="SBBYE",
        gsis_id=None,
    )

    result = quality.check_null_rates(
        df,
        spec,
    )

    assert not result.passed
    assert result.severity == "error"

    assert "2024:gsis_id" in result.data["key_nulls"]
    assert "2024:week" not in result.data["key_nulls"]