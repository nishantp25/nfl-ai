"""Regression tests for live-provider coverage and schema verification.

A reachable URL is evidence that an asset exists. It is not evidence that the
asset contains the expected schema or that its internal historical coverage is
what the manifest claims.

These tests keep those concepts separate.
"""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from nfldl import verify
from nfldl.manifest import Coverage, DatasetSpec


def _spec(
    name: str = "demo",
    *,
    start: int | None = 2020,
    end: int | None = 2022,
    primary_key: tuple[str, ...] = ("season", "row_id"),
    timestamp_field: str | None = "updated_at",
    partition_by: tuple[str, ...] = ("season",),
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        provider="test-provider",
        tier="nflverse",
        access="test",
        declared=Coverage(start, end),
        verified=Coverage(start, end),
        update_frequency="test",
        local_storage_path="data/raw/test/{season}.parquet",
        primary_key=primary_key,
        timestamp_field=timestamp_field,
        partition_by=partition_by,
        required=True,
        role="verification test fixture",
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        target_seasons=[2020, 2021, 2022],
        max_workers=1,
    )


def _install_common_fakes(
    monkeypatch,
    spec: DatasetSpec,
    adapter: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        verify,
        "get_manifest",
        lambda: {spec.name: spec},
    )

    monkeypatch.setattr(
        verify,
        "get_adapter",
        lambda _name: adapter,
    )

    monkeypatch.setattr(
        verify,
        "get_settings",
        _settings,
    )

    monkeypatch.setattr(
        verify,
        "nfl_season_of",
        lambda: 2022,
    )


def test_head_only_seasonal_probe_does_not_claim_verified_coverage(
    monkeypatch,
):
    """HEAD success proves reachability, not schema or content coverage."""
    spec = _spec()

    def fetch_must_not_run(**_kwargs):
        raise AssertionError(
            "sample=False must not download the dataset"
        )

    adapter = SimpleNamespace(
        asset_url=lambda **_kwargs: "https://example.test/demo.parquet",
        needs_local=(),
        seasonal_assets=True,
        variant_assets=False,
        fetch=fetch_must_not_run,
    )

    _install_common_fakes(
        monkeypatch,
        spec,
        adapter,
    )

    monkeypatch.setattr(
        verify,
        "_probe_seasons",
        lambda _spec, _current: (
            [2020, 2021, 2022],
            2020,
        ),
    )

    probe = verify.probe_dataset(
        spec.name,
        sample=False,
    )

    assert probe.seasons_present == [
        2020,
        2021,
        2022,
    ]

    assert probe.asset_reachable is True
    assert probe.asset_start == 2020
    assert probe.asset_end == 2022

    assert probe.observed_start is None
    assert probe.observed_end is None

    assert probe.schema_verified is False

    assert probe.verified_start is None
    assert probe.verified_end is None


def test_sampled_seasonal_probe_verifies_schema_without_claiming_full_range(
    monkeypatch,
):
    """One downloaded season validates that season, not every HEAD-visible year."""
    spec = _spec()

    adapter = SimpleNamespace(
        asset_url=lambda **_kwargs: "https://example.test/demo.parquet",
        needs_local=(),
        seasonal_assets=True,
        variant_assets=False,
        fetch=lambda **_kwargs: pl.DataFrame(
            {
                "season": [
                    2022,
                    2022,
                ],
                "row_id": [
                    1,
                    2,
                ],
                "updated_at": [
                    "2022-09-01",
                    "2022-09-02",
                ],
            }
        ),
    )

    _install_common_fakes(
        monkeypatch,
        spec,
        adapter,
    )

    monkeypatch.setattr(
        verify,
        "_probe_seasons",
        lambda _spec, _current: (
            [2020, 2021, 2022],
            2020,
        ),
    )

    probe = verify.probe_dataset(
        spec.name,
        sample=True,
    )

    assert probe.asset_reachable is True
    assert probe.asset_start == 2020
    assert probe.asset_end == 2022

    assert probe.sampled_season == 2022
    assert probe.sampled_rows == 2

    assert probe.observed_start == 2022
    assert probe.observed_end == 2022

    assert probe.schema_verified is True

    # Only 2022 was downloaded and inspected. HEAD checks for 2020 and 2021
    # do not prove that those assets share the verified schema.
    assert probe.verified_start is None
    assert probe.verified_end is None


def test_schema_verification_rejects_missing_timestamp_field(
    monkeypatch,
):
    """A declared timestamp column is part of the schema contract."""
    spec = _spec()

    adapter = SimpleNamespace(
        asset_url=lambda **_kwargs: "https://example.test/demo.parquet",
        needs_local=(),
        seasonal_assets=True,
        variant_assets=False,
        fetch=lambda **_kwargs: pl.DataFrame(
            {
                "season": [
                    2022,
                    2022,
                ],
                "row_id": [
                    1,
                    2,
                ],
            }
        ),
    )

    _install_common_fakes(
        monkeypatch,
        spec,
        adapter,
    )

    monkeypatch.setattr(
        verify,
        "_probe_seasons",
        lambda _spec, _current: (
            [2022],
            2020,
        ),
    )

    probe = verify.probe_dataset(
        spec.name,
        sample=True,
    )

    assert probe.schema_verified is False
    assert probe.ok is False

    assert any(
        "timestamp" in note.lower()
        for note in probe.notes
    )


def test_sampled_single_file_can_establish_full_internal_range(
    monkeypatch,
):
    """Downloading an all-history file can reveal its real internal coverage."""
    spec = _spec(
        name="all_history",
        start=2011,
        end=2025,
        primary_key=(
            "draft_year",
            "row_id",
        ),
        timestamp_field=None,
        partition_by=(),
    )

    adapter = SimpleNamespace(
        asset_url=lambda **_kwargs: (
            "https://example.test/all-history.parquet"
        ),
        needs_local=(),
        seasonal_assets=False,
        variant_assets=False,
        fetch=lambda **_kwargs: pl.DataFrame(
            {
                "draft_year": [
                    2012,
                    2013,
                    2014,
                ],
                "row_id": [
                    1,
                    2,
                    3,
                ],
            }
        ),
    )

    _install_common_fakes(
        monkeypatch,
        spec,
        adapter,
    )

    monkeypatch.setattr(
        verify,
        "head_ok",
        lambda _url: True,
    )

    probe = verify.probe_dataset(
        spec.name,
        sample=True,
    )

    assert probe.asset_reachable is True

    assert probe.sampled_rows == 3

    assert probe.sampled_columns == [
        "draft_year",
        "row_id",
    ]

    assert probe.seasons_present == [
        2012,
        2013,
        2014,
    ]

    assert probe.observed_start == 2012
    assert probe.observed_end == 2014

    assert probe.schema_verified is True

    # This is one all-history object, so inspecting that object can establish
    # the historical range actually contained inside it.
    assert probe.verified_start == 2012
    assert probe.verified_end == 2014


def test_write_back_skips_unverified_coverage(
    tmp_path,
):
    """Unverified HEAD-derived numbers must never overwrite committed YAML."""
    path = tmp_path / "data_sources.yaml"

    original = (
        'manifest:\n'
        '  verified_on: "2026-01-01"\n'
        'datasets:\n'
        '  demo:\n'
        '    verified_start: 2011\n'
        '    verified_end: 2025\n'
    )

    path.write_text(
        original,
        encoding="utf-8",
    )

    report = {
        "generated_at": "2026-09-02T18:00:00+00:00",
        "datasets": {
            "demo": {
                "schema_verified": False,
                "verified_start": 2020,
                "verified_end": 2022,
            }
        },
    }

    verify.write_back_to_manifest(
        report,
        path,
    )

    assert path.read_text(
        encoding="utf-8"
    ) == original


def test_write_back_accepts_genuinely_verified_coverage(
    tmp_path,
):
    """Content-backed verified ranges may be written into the manifest."""
    path = tmp_path / "data_sources.yaml"

    path.write_text(
        (
            'manifest:\n'
            '  verified_on: "2026-01-01"\n'
            'datasets:\n'
            '  demo:\n'
            '    verified_start: 2011\n'
            '    verified_end: 2025\n'
        ),
        encoding="utf-8",
    )

    report = {
        "generated_at": "2026-09-02T18:00:00+00:00",
        "datasets": {
            "demo": {
                "schema_verified": True,
                "verified_start": 2012,
                "verified_end": 2014,
            }
        },
    }

    verify.write_back_to_manifest(
        report,
        path,
    )

    updated = path.read_text(
        encoding="utf-8"
    )

    assert 'verified_on: "2026-09-02"' in updated
    assert "verified_start: 2012" in updated
    assert "verified_end: 2014" in updated