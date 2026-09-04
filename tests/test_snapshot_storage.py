"""Regression tests for append-only snapshot datasets."""

from datetime import datetime, timezone
from types import SimpleNamespace

import polars as pl

from nfldl import ingest, storage
from nfldl.manifest import Coverage, DatasetSpec


def _snapshot_spec(*, append: bool = True) -> DatasetSpec:
    return DatasetSpec(
        name="snapshot_test",
        provider="test_provider",
        tier="reputable_public",
        access="test",
        declared=Coverage(2026, 2026),
        verified=Coverage(2026, 2026),
        update_frequency="test",
        local_storage_path=(
            "data/raw/snapshot_test/season={season}/part.parquet"
        ),
        primary_key=(
            "game_id",
            "retrieved_at_utc",
        ),
        timestamp_field="retrieved_at_utc",
        partition_by=("season",),
        required=False,
        raw=(
            {"storage_mode": "append"}
            if append
            else {}
        ),
    )


def _utc(hour: int) -> datetime:
    return datetime(
        2026,
        9,
        3,
        hour,
        tzinfo=timezone.utc,
    )


def test_append_partition_preserves_old_rows_and_adds_new_snapshots(
    tmp_path,
    monkeypatch,
):
    spec = _snapshot_spec()

    monkeypatch.setattr(
        storage,
        "PROVENANCE_LOG_PATH",
        tmp_path / "reports" / "ingestion_log.jsonl",
    )

    first = pl.DataFrame(
        {
            "game_id": ["g1"],
            "retrieved_at_utc": [_utc(12)],
            "temperature_c": [10.0],
            "season": [2026],
        }
    )

    storage.write_partition(
        first,
        spec,
        root=tmp_path,
        season=2026,
    )

    # The first row deliberately reuses the old primary key with a
    # conflicting value. An append-only lake must retain the version
    # that was already stored rather than rewrite history.
    second = pl.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "retrieved_at_utc": [
                _utc(12),
                _utc(13),
            ],
            "temperature_c": [
                999.0,
                20.0,
            ],
            "season": [
                2026,
                2026,
            ],
        }
    )

    storage.write_partition(
        second,
        spec,
        root=tmp_path,
        season=2026,
    )

    dest = spec.storage_path(
        tmp_path,
        season=2026,
    )

    out = (
        pl.read_parquet(dest)
        .sort("retrieved_at_utc")
    )

    assert out.height == 2
    assert out.get_column("temperature_c").to_list() == [
        10.0,
        20.0,
    ]

    # Each row keeps the ingest provenance from the run that first
    # introduced that immutable snapshot.
    assert (
        out.get_column("_ingested_at")
        .n_unique()
        == 2
    )


def test_append_dataset_is_not_skipped_when_partition_exists(
    monkeypatch,
):
    spec = _snapshot_spec()
    fetched = {"called": False}
    written = {"called": False}

    def fetch(**_):
        fetched["called"] = True
        return pl.DataFrame(
            {
                "game_id": ["g1"],
                "retrieved_at_utc": [_utc(12)],
            }
        )

    def fake_write(
        df,
        _spec,
        **_,
    ):
        written["called"] = True
        return SimpleNamespace(
            rows=df.height,
        )

    monkeypatch.setattr(
        ingest,
        "get_manifest",
        lambda: {"snapshot_test": spec},
    )
    monkeypatch.setattr(
        ingest,
        "get_adapter",
        lambda _name: SimpleNamespace(
            fetch=fetch,
        ),
    )
    monkeypatch.setattr(
        ingest,
        "_seasons_for",
        lambda _spec, _requested: (
            [2026],
            [],
        ),
    )
    monkeypatch.setattr(
        ingest,
        "_partitions_for",
        lambda _spec, _seasons: [
            {"season": 2026}
        ],
    )
    monkeypatch.setattr(
        ingest,
        "partition_exists",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        ingest,
        "write_partition",
        fake_write,
    )

    result = ingest.ingest_dataset(
        "snapshot_test",
        seasons=[2026],
    )

    assert fetched["called"]
    assert written["called"]
    assert result[0].status == "ingested"


def test_normal_dataset_still_skips_existing_partition(
    monkeypatch,
):
    spec = _snapshot_spec(
        append=False,
    )
    fetched = {"called": False}

    def fetch(**_):
        fetched["called"] = True
        return pl.DataFrame(
            {
                "game_id": ["g1"],
                "retrieved_at_utc": [_utc(12)],
            }
        )

    monkeypatch.setattr(
        ingest,
        "get_manifest",
        lambda: {"snapshot_test": spec},
    )
    monkeypatch.setattr(
        ingest,
        "get_adapter",
        lambda _name: SimpleNamespace(
            fetch=fetch,
        ),
    )
    monkeypatch.setattr(
        ingest,
        "_seasons_for",
        lambda _spec, _requested: (
            [2026],
            [],
        ),
    )
    monkeypatch.setattr(
        ingest,
        "_partitions_for",
        lambda _spec, _seasons: [
            {"season": 2026}
        ],
    )
    monkeypatch.setattr(
        ingest,
        "partition_exists",
        lambda *_args, **_kwargs: True,
    )

    result = ingest.ingest_dataset(
        "snapshot_test",
        seasons=[2026],
    )

    assert not fetched["called"]
    assert result[0].status == "skipped_exists"
