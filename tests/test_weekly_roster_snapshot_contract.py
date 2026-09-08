"""Contract tests for point-in-time live weekly-roster snapshots.

Historical ``weekly_rosters`` remain retrospective week-grained observations
because nflverse does not provide a publication timestamp for those rows.

``weekly_roster_snapshots`` records roster rows actually observed by this
project at a known retrieval time.  It therefore provides the PIT-safe live
counterpart without inventing historical knowledge timestamps or player IDs.
"""

from datetime import datetime, timezone

import polars as pl

from nfldl import pit
from nfldl.manifest import load_manifest
from nfldl.registry import REGISTRY, resolve_order


def test_weekly_roster_snapshots_are_a_separate_live_dataset():
    manifest = load_manifest()

    assert "weekly_roster_snapshots" in manifest.names()

    spec = manifest["weekly_roster_snapshots"]

    assert not spec.required
    assert spec.primary_key == (
        "season",
        "week",
        "team",
        "source_row_sha256",
        "retrieved_at_utc",
    )
    assert spec.timestamp_field == "retrieved_at_utc"
    assert spec.raw.get("storage_mode") == "append"
    assert spec.raw.get("point_in_time") == "retrieval"

    # Collection begins when this project starts recording live observations;
    # retrospective nflverse week labels do not certify old PIT coverage.
    assert spec.verified.start is None
    assert spec.verified.end is None


def test_weekly_roster_snapshot_is_explicitly_pregame_by_retrieval_time():
    assert pit.DATASET_AVAILABILITY["weekly_roster_snapshots"] == "pregame"


def test_weekly_roster_snapshot_adapter_refreshes_weekly_rosters_first():
    assert "weekly_roster_snapshots" in REGISTRY

    adapter = REGISTRY["weekly_roster_snapshots"]

    assert adapter.needs_local == ("weekly_rosters",)

    order = resolve_order(["weekly_roster_snapshots"])

    assert order.index("weekly_rosters") < order.index(
        "weekly_roster_snapshots"
    )


def test_live_builder_keeps_latest_week_and_preserves_unidentified_players(
    monkeypatch,
):
    from nfldl.sources import weekly_roster_snapshots

    old_ingest = datetime(
        2026,
        9,
        7,
        12,
        0,
        tzinfo=timezone.utc,
    )
    retrieved = datetime(
        2026,
        9,
        8,
        9,
        30,
        tzinfo=timezone.utc,
    )

    source = pl.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "week": [1, 2, 2, 2],
            "game_type": ["REG", "REG", "REG", "REG"],
            "team": ["NYJ", "NYJ", "NYJ", "NYJ"],
            "full_name": [
                "Old Week Player",
                "Known Player",
                "Al-Jay Henderson",
                "Al-Jay Henderson",
            ],
            "position": ["RB", "WR", "RB", "RB"],
            "status": ["ACT", "ACT", "DEV", "DEV"],
            "gsis_id": [
                "00-0000001",
                "00-0000002",
                None,
                None,
            ],
            "gsis_it_id": [
                "1",
                "2",
                "63414",
                "63414",
            ],
            "_ingested_at": [
                old_ingest,
                old_ingest,
                old_ingest,
                old_ingest,
            ],
            "_source_tier": [
                "nflverse",
                "nflverse",
                "nflverse",
                "nflverse",
            ],
            "_provider": [
                "nflverse",
                "nflverse",
                "nflverse",
                "nflverse",
            ],
        }
    )

    monkeypatch.setattr(
        weekly_roster_snapshots.storage,
        "read_dataset",
        lambda *args, **kwargs: source,
    )
    monkeypatch.setattr(
        weekly_roster_snapshots,
        "_utc_now",
        lambda: retrieved,
    )

    out = weekly_roster_snapshots.build(2026)

    # Week 1 is historical relative to the latest available source snapshot.
    assert out.get_column("week").unique().to_list() == [2]

    # The exact duplicate DEV row is removed, but the player is retained even
    # though no GSIS player ID is available.
    assert out.height == 2
    assert out.filter(
        pl.col("gsis_id").is_null()
    ).height == 1

    assert out.get_column(
        "retrieved_at_utc"
    ).unique().to_list() == [retrieved]

    fingerprints = out.get_column(
        "source_row_sha256"
    ).to_list()

    assert len(set(fingerprints)) == out.height
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in fingerprints
    )

    # Generic ingest bookkeeping from the source partition is deliberately not
    # copied into the observed row identity or exposed as knowledge time.
    assert "_ingested_at" not in out.columns
    assert "_source_tier" not in out.columns
    assert "_provider" not in out.columns


def test_source_row_fingerprint_ignores_generic_ingest_bookkeeping():
    from nfldl.sources import weekly_roster_snapshots

    first = pl.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "team": ["NYJ"],
            "full_name": ["Al-Jay Henderson"],
            "position": ["RB"],
            "status": ["DEV"],
            "gsis_id": [None],
            "gsis_it_id": ["63414"],
            "_ingested_at": [
                datetime(
                    2026,
                    9,
                    8,
                    8,
                    0,
                    tzinfo=timezone.utc,
                )
            ],
            "_source_tier": ["nflverse"],
            "_provider": ["nflverse"],
        }
    )

    second = first.with_columns(
        pl.lit(
            datetime(
                2026,
                9,
                8,
                9,
                0,
                tzinfo=timezone.utc,
            )
        ).alias("_ingested_at")
    )

    first_hash = weekly_roster_snapshots._fingerprint_source_rows(
        first
    ).get_column("source_row_sha256")[0]

    second_hash = weekly_roster_snapshots._fingerprint_source_rows(
        second
    ).get_column("source_row_sha256")[0]

    assert first_hash == second_hash
