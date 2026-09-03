"""Verifier regression tests for season-aware schema contracts.

The manifest can now declare schema epochs, but live verification must actually
use the contract that applies to the sampled season.

Confirmed nflverse boundaries:

* injuries
  - through 2024: ``date_modified`` is required
  - 2025+: no ``date_modified`` field

* depth_charts
  - through 2024: legacy weekly snapshot key
  - 2025+: timestamped feed with a different primary key
"""

from __future__ import annotations

import polars as pl

from nfldl import verify
from nfldl.manifest import DatasetSpec, load_manifest


def _probe(spec: DatasetSpec) -> verify.DatasetProbe:
    """Build the minimum probe object needed for schema validation."""
    return verify.DatasetProbe(
        dataset=spec.name,
        provider=spec.provider,
        declared_start=spec.declared.start,
        declared_end=spec.declared.end,
    )


def test_verify_2025_injuries_uses_new_timestamp_contract():
    """2025 injuries must not require the removed date_modified field."""
    spec = load_manifest()["injuries"]

    df = pl.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 1],
            "team": ["KC", "BUF"],
            "gsis_id": ["p1", "p2"],
            "season_type": ["REG", "REG"],
            "report_status": ["Questionable", "Out"],
        }
    )

    probe = _probe(spec)

    valid = verify._check_schema(
        df,
        spec,
        probe,
        season=2025,
    )

    assert valid
    assert probe.ok
    assert not any(
        "date_modified" in note
        for note in probe.notes
    )


def test_verify_2024_injuries_still_requires_date_modified():
    """The old injury contract must remain strict through 2024."""
    spec = load_manifest()["injuries"]

    df = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "team": ["KC"],
            "gsis_id": ["p1"],
            # Deliberately missing date_modified.
        }
    )

    probe = _probe(spec)

    valid = verify._check_schema(
        df,
        spec,
        probe,
        season=2024,
    )

    assert not valid
    assert not probe.ok
    assert any(
        "date_modified" in note
        for note in probe.notes
    )


def test_verify_2025_depth_charts_uses_new_primary_key():
    """The verifier must validate the timestamped 2025 depth-chart key."""
    spec = load_manifest()["depth_charts"]

    df = pl.DataFrame(
        {
            "dt": [
                "2025-08-01T12:00:00",
                "2025-08-01T12:00:00",
            ],
            "team": ["KC", "KC"],
            "gsis_id": ["p1", "p1"],
            "pos_id": [1, 1],
            "pos_slot": [1, 2],
            "pos_rank": [1, 2],
        }
    )

    probe = _probe(spec)

    valid = verify._check_schema(
        df,
        spec,
        probe,
        season=2025,
    )

    assert valid
    assert probe.ok


def test_verify_2024_depth_charts_does_not_require_dt():
    """Legacy depth charts use the weekly key and have no knowledge timestamp."""
    spec = load_manifest()["depth_charts"]

    df = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "club_code": ["KC", "KC"],
            "gsis_id": ["p1", "p2"],
            "depth_position": ["QB1", "QB2"],
            "game_type": ["REG", "REG"],
            "depth_team": ["1", "2"],
            "formation": ["Offense", "Offense"],
        }
    )

    probe = _probe(spec)

    valid = verify._check_schema(
        df,
        spec,
        probe,
        season=2024,
    )

    assert valid
    assert probe.ok
    assert not any(
        "timestamp" in note.lower()
        for note in probe.notes
    )