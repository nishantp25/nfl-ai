"""Live injury observations; retrieval time does not establish historical availability."""
from datetime import datetime, timezone
from typing import Any
import polars as pl
from .. import storage
from ..config import RAW_DIR
from ..manifest import get_manifest
from .weekly_roster_snapshots import _fingerprint_source_rows, _strip_ingest_bookkeeping

def build(season: int, **_: Any) -> pl.DataFrame:
    source = storage.read_dataset(
        get_manifest()["injuries"], root=RAW_DIR, seasons=[season]
    )
    required = {"season", "week", "team", "gsis_id", "report_status"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Injury snapshot source lacks columns: {sorted(missing)}")
    if source.is_empty():
        return pl.DataFrame()
    if source.filter(pl.col("season").is_null() | (pl.col("season") != season)).height:
        raise ValueError("Injury source contains conflicting season values")
    snapshot = _strip_ingest_bookkeeping(source).unique(maintain_order=True)
    snapshot = _fingerprint_source_rows(snapshot)
    retrieved = datetime.now(timezone.utc)
    return snapshot.with_columns(
        pl.lit(retrieved).cast(pl.Datetime("us", "UTC")).alias("retrieved_at_utc")
    )
