"""Regression tests for non-seasonal ingestion coverage semantics."""

from nfldl.ingest import _partitions_for, _seasons_for
from nfldl.manifest import get_manifest


def test_nonseasonal_dataset_ignores_requested_season_coverage():
    spec = get_manifest()["stadiums"]

    assert spec.is_partitioned_by_season is False

    available, unavailable = _seasons_for(
        spec,
        [2026],
    )

    assert available == []
    assert unavailable == []

    assert _partitions_for(
        spec,
        available,
    ) == [{}]
