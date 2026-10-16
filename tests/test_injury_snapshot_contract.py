from datetime import datetime, timedelta, timezone
import polars as pl
import pytest
from nfldl import pit
from nfldl.manifest import load_manifest
from nfldl.registry import get_adapter, resolve_order
from nfldl.sources import injury_snapshots

def source():
    return pl.DataFrame({
        "season": [2026, 2026, 2026],
        "week": [5, 6, 6],
        "team": ["KC", "BUF", "BUF"],
        "gsis_id": ["00-example", None, None],
        "report_status": ["Out", "Questionable", "Questionable"],
        "_ingested_at": [datetime(2026, 9, 1, tzinfo=timezone.utc)] * 3,
    })

def test_snapshot_contract_and_dependency():
    spec = load_manifest()["injury_snapshots"]
    assert spec.raw["storage_mode"] == "append"
    assert spec.timestamp_field == "retrieved_at_utc"
    assert spec.verified.start is None
    assert spec.verified.end is None
    assert "retrieved_at_utc" in spec.primary_key
    assert get_adapter("injury_snapshots").needs_local == ("injuries",)
    assert resolve_order(["injury_snapshots"]) == ["injuries", "injury_snapshots"]
    assert pit.DATASET_AVAILABILITY["injury_snapshots"] == "pregame"

def test_actual_observation_time_and_identity_preservation(monkeypatch):
    monkeypatch.setattr(injury_snapshots.storage, "read_dataset", lambda *a, **k: source())
    before = datetime.now(timezone.utc)
    out = injury_snapshots.build(2026)
    after = datetime.now(timezone.utc)
    assert out.height == 2
    assert out["week"].to_list() == [5, 6]
    assert out["gsis_id"].null_count() == 1
    assert "_ingested_at" not in out.columns
    assert all(before <= t <= after for t in out["retrieved_at_utc"])
    assert out["source_row_sha256"].n_unique() == 2

def test_changed_status_changes_fingerprint(monkeypatch):
    original = source()
    monkeypatch.setattr(injury_snapshots.storage, "read_dataset", lambda *a, **k: original)
    first = injury_snapshots.build(2026)
    changed = original.with_columns(pl.lit("Doubtful").alias("report_status"))
    monkeypatch.setattr(injury_snapshots.storage, "read_dataset", lambda *a, **k: changed)
    second = injury_snapshots.build(2026)
    assert set(first["source_row_sha256"]).isdisjoint(set(second["source_row_sha256"]))

def test_wrong_season_rejected(monkeypatch):
    wrong = source().with_columns(pl.lit(2025).alias("season"))
    monkeypatch.setattr(injury_snapshots.storage, "read_dataset", lambda *a, **k: wrong)
    with pytest.raises(ValueError, match="conflicting season"):
        injury_snapshots.build(2026)

def test_missing_status_rejected(monkeypatch):
    monkeypatch.setattr(injury_snapshots.storage, "read_dataset",
                        lambda *a, **k: source().drop("report_status"))
    with pytest.raises(ValueError, match="lacks columns"):
        injury_snapshots.build(2026)

def test_observation_at_kickoff_is_excluded():
    cutoff = datetime(2026, 10, 18, 17, tzinfo=timezone.utc)
    observations = pl.DataFrame({
        "team": ["KC", "KC", "KC"],
        "kickoff_utc": [cutoff] * 3,
        "retrieved_at_utc": [cutoff - timedelta(hours=1), cutoff, cutoff + timedelta(hours=1)],
        "report_status": ["Questionable", "Out", "Doubtful"],
    })
    known = pit.knowledge_time(observations, "injury_snapshots", report_col="retrieved_at_utc")
    left = pl.DataFrame({"team": ["KC"], "kickoff_utc": [cutoff]})
    joined = pit.asof_join(left, known.drop("kickoff_utc"), by=["team"])
    assert joined["report_status"][0] == "Questionable"
    pit.assert_no_future_rows(joined)
