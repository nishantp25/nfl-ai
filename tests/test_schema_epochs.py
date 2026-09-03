"""Regression tests for season-aware schema contracts.

nflverse occasionally changes a dataset's schema at a season boundary. A
single global primary key or timestamp contract is therefore not always
truthful.

Confirmed live examples:

* injuries
  - 2009-2024: ``date_modified`` exists.
  - 2025+: ``date_modified`` is absent.

* depth_charts
  - 2001-2024: weekly snapshot schema.
  - 2025+: timestamped depth updates with a materially different key.

These tests require the manifest, ingestion, and quality layers to understand
those boundaries explicitly instead of treating schema drift as an arbitrary
collection of nullable columns.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import yaml

from nfldl import ingest, quality
from nfldl.config import MANIFEST_PATH
from nfldl.manifest import ManifestError, load_manifest


def _write_manifest_with_epochs(
    tmp_path: Path,
    dataset: str,
    epochs: list[dict[str, object]],
) -> Path:
    """Copy the project manifest and replace one dataset's schema epochs."""
    payload = yaml.safe_load(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    payload["datasets"][dataset]["schema_epochs"] = epochs

    path = tmp_path / "data_sources.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Manifest schema resolution
# ---------------------------------------------------------------------------


def test_injury_schema_contract_changes_in_2025():
    """The 2025 injury source no longer carries date_modified."""
    spec = load_manifest()["injuries"]

    old = spec.schema_for_season(2024)
    new = spec.schema_for_season(2025)

    assert old.primary_key == (
        "season",
        "week",
        "team",
        "gsis_id",
    )
    assert old.timestamp_field == "date_modified"

    assert new.primary_key == (
        "season",
        "week",
        "team",
        "gsis_id",
    )
    assert new.timestamp_field is None


def test_depth_chart_schema_contract_changes_in_2025():
    """The 2025 depth-chart feed is a different schema, not minor drift."""
    spec = load_manifest()["depth_charts"]

    old = spec.schema_for_season(2024)
    new = spec.schema_for_season(2025)

    assert old.primary_key == (
        "season",
        "week",
        "club_code",
        "gsis_id",
        "depth_position",
        "game_type",
        "depth_team",
        "formation",
    )
    assert old.timestamp_field is None

    assert new.primary_key == (
        "dt",
        "team",
        "gsis_id",
        "pos_id",
        "pos_slot",
        "pos_rank",
    )
    assert new.timestamp_field == "dt"


def test_dataset_without_epochs_keeps_default_schema_contract():
    """Most datasets should not need duplicated schema configuration."""
    spec = load_manifest()["schedules"]

    contract = spec.schema_for_season(2024)

    assert contract.primary_key == spec.primary_key
    assert contract.timestamp_field == spec.timestamp_field


def test_schema_epochs_may_not_overlap(tmp_path):
    """A season must never match two competing schema definitions."""
    path = _write_manifest_with_epochs(
        tmp_path,
        "injuries",
        [
            {
                "start": 2009,
                "end": 2024,
                "primary_key": [
                    "season",
                    "week",
                    "team",
                    "gsis_id",
                ],
                "timestamp_field": "date_modified",
            },
            {
                "start": 2024,
                "end": None,
                "primary_key": [
                    "season",
                    "week",
                    "team",
                    "gsis_id",
                ],
                "timestamp_field": None,
            },
        ],
    )

    with pytest.raises(
        ManifestError,
        match="overlap",
    ):
        load_manifest(path)


def test_schema_epochs_may_not_leave_internal_gaps(tmp_path):
    """Declared coverage cannot silently contain a season with no contract."""
    path = _write_manifest_with_epochs(
        tmp_path,
        "injuries",
        [
            {
                "start": 2009,
                "end": 2023,
                "primary_key": [
                    "season",
                    "week",
                    "team",
                    "gsis_id",
                ],
                "timestamp_field": "date_modified",
            },
            {
                "start": 2025,
                "end": None,
                "primary_key": [
                    "season",
                    "week",
                    "team",
                    "gsis_id",
                ],
                "timestamp_field": None,
            },
        ],
    )

    with pytest.raises(
        ManifestError,
        match="gap",
    ):
        load_manifest(path)


# ---------------------------------------------------------------------------
# Quality checks across multiple schema epochs
# ---------------------------------------------------------------------------


def _mixed_depth_chart_frame() -> pl.DataFrame:
    """Small frame containing both sides of the 2025 schema boundary."""
    old = pl.DataFrame(
        {
            "season": [
                2024,
                2024,
            ],
            "week": [
                1,
                1,
            ],
            "club_code": [
                "KC",
                "KC",
            ],
            "gsis_id": [
                "old-1",
                "old-2",
            ],
            "depth_position": [
                "QB1",
                "QB2",
            ],
            "game_type": [
                "REG",
                "REG",
            ],
            "depth_team": [
                "1",
                "2",
            ],
            "formation": [
                "Offense",
                "Offense",
            ],
        }
    )

    # The same player may legitimately occupy multiple position slots in the
    # new feed. The complete 2025 key distinguishes these records.
    new = pl.DataFrame(
        {
            "season": [
                2025,
                2025,
            ],
            "dt": [
                "2025-08-01T12:00:00",
                "2025-08-01T12:00:00",
            ],
            "team": [
                "KC",
                "KC",
            ],
            "gsis_id": [
                "new-1",
                "new-1",
            ],
            "pos_id": [
                1,
                1,
            ],
            "pos_slot": [
                1,
                2,
            ],
            "pos_rank": [
                1,
                2,
            ],
        }
    )

    return pl.concat(
        [
            old,
            new,
        ],
        how="diagonal_relaxed",
    )


def test_primary_key_check_uses_the_contract_for_each_season():
    """Null legacy fields in 2025 must not create fake duplicate keys."""
    spec = load_manifest()["depth_charts"]
    df = _mixed_depth_chart_frame()

    result = quality.check_primary_key(
        df,
        spec,
    )

    assert result.passed
    assert result.severity == "info"


def test_null_check_ignores_inactive_epoch_key_columns():
    """A field absent by design in another epoch is not a corrupt null key."""
    spec = load_manifest()["depth_charts"]
    df = _mixed_depth_chart_frame()

    result = quality.check_null_rates(
        df,
        spec,
    )

    assert result.passed
    assert result.severity != "error"


# ---------------------------------------------------------------------------
# Partition metadata
# ---------------------------------------------------------------------------


def test_partition_season_is_restored_when_provider_omits_it():
    """Partition identity must survive even when upstream drops season."""
    df = pl.DataFrame(
        {
            "dt": [
                "2025-08-01T12:00:00",
                "2025-08-02T12:00:00",
            ],
            "team": [
                "KC",
                "BUF",
            ],
        }
    )

    restored = ingest.attach_partition_keys(
        df,
        {
            "season": 2025,
        },
    )

    assert restored.get_column("season").to_list() == [
        2025,
        2025,
    ]


def test_partition_metadata_does_not_overwrite_conflicting_provider_value():
    """A path claiming 2025 must not silently rewrite rows labelled 2024."""
    df = pl.DataFrame(
        {
            "season": [
                2024,
            ],
            "team": [
                "KC",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="season",
    ):
        ingest.attach_partition_keys(
            df,
            {
                "season": 2025,
            },
        )


def test_matching_provider_partition_value_is_preserved():
    """Existing trustworthy partition columns should remain valid."""
    df = pl.DataFrame(
        {
            "season": [
                2025,
                2025,
            ],
            "team": [
                "KC",
                "BUF",
            ],
        }
    )

    restored = ingest.attach_partition_keys(
        df,
        {
            "season": 2025,
        },
    )

    assert restored.get_column("season").to_list() == [
        2025,
        2025,
    ]
    assert restored.get_column("team").to_list() == [
        "KC",
        "BUF",
    ]