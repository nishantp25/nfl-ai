"""Point-in-time snapshots of live nflverse weekly-roster observations.

The historical ``weekly_rosters`` dataset is intentionally *not* treated as a
same-week pregame source because nflverse rows do not contain a publication or
knowledge timestamp.

This module provides the live counterpart.  It snapshots the latest available
weekly-roster state after the project's ordinary ``weekly_rosters`` ingest has
completed and records ``retrieved_at_utc`` as the conservative knowledge time.

Player identity is not invented here.  ``source_row_sha256`` identifies the
exact source row that was observed, including rows whose ``gsis_id`` is null.
Canonical player/entity resolution remains a separate downstream concern.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Any

import polars as pl

from .. import storage
from ..config import RAW_DIR
from ..manifest import get_manifest

_FINGERPRINT_EXCLUDED = frozenset(
    {
        *storage.INGEST_COLUMNS,
        "retrieved_at_utc",
        "source_row_sha256",
    }
)


def _utc_now() -> datetime:
    """Current UTC instant, isolated so snapshot timing is testable."""
    return datetime.now(timezone.utc)


def _json_default(value: Any) -> str:
    """Canonical JSON fallback for source scalar values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    return str(value)


def _fingerprint_source_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Attach a deterministic SHA-256 fingerprint to each observed source row.

    Generic ingest bookkeeping and live-snapshot metadata are deliberately
    excluded.  Column names are sorted before serialization so a harmless
    source-column reordering does not change row identity.
    """
    source_columns = sorted(
        column
        for column in df.columns
        if column not in _FINGERPRINT_EXCLUDED
    )

    if not source_columns:
        raise ValueError(
            "weekly roster snapshot has no source columns to fingerprint"
        )

    payload = df.select(source_columns)

    hashes: list[str] = []

    for row in payload.iter_rows(named=True):
        canonical = json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_json_default,
        )
        hashes.append(
            hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()
        )

    return df.with_columns(
        pl.Series(
            "source_row_sha256",
            hashes,
            dtype=pl.String,
        )
    )


def _strip_ingest_bookkeeping(df: pl.DataFrame) -> pl.DataFrame:
    """Remove processing metadata that is not part of roster knowledge."""
    drop = [
        column
        for column in storage.INGEST_COLUMNS
        if column in df.columns
    ]

    return df.drop(drop) if drop else df


def build(
    season: int,
    **_: Any,
) -> pl.DataFrame:
    """Build one immutable observation of the latest available roster week.

    ``weekly_rosters`` must be refreshed first by the registry dependency.
    Retrieval time is captured only after the local source snapshot has been
    read and transformed, making it a conservative knowledge timestamp.

    Only the latest available week is emitted. Exact duplicate source rows are
    removed before fingerprinting, but unidentified players are retained.
    """
    mf = get_manifest()

    source = storage.read_dataset(
        mf["weekly_rosters"],
        root=RAW_DIR,
        seasons=[season],
    )

    if isinstance(source, pl.LazyFrame):
        source = source.collect()

    if source.is_empty():
        return pl.DataFrame()

    required = {
        "season",
        "week",
        "team",
    }
    missing = sorted(required - set(source.columns))

    if missing:
        raise ValueError(
            "weekly_rosters source is missing required live-snapshot "
            f"columns {missing}"
        )

    latest_week = source.get_column("week").drop_nulls().max()

    if latest_week is None:
        return pl.DataFrame()

    snapshot = (
        source
        .filter(pl.col("week") == latest_week)
        .pipe(_strip_ingest_bookkeeping)
        .unique(
            maintain_order=True
        )
    )

    if snapshot.is_empty():
        return pl.DataFrame()

    snapshot = _fingerprint_source_rows(
        snapshot
    )

    # Capture the knowledge timestamp only after the complete source snapshot
    # has been read, filtered, deduplicated and fingerprinted.
    retrieved_at_utc = _utc_now()

    if retrieved_at_utc.tzinfo is None:
        raise ValueError(
            "retrieved_at_utc must be timezone-aware"
        )

    retrieved_at_utc = retrieved_at_utc.astimezone(
        timezone.utc
    )

    return snapshot.with_columns(
        pl.lit(retrieved_at_utc)
        .cast(
            pl.Datetime(
                "us",
                "UTC",
            )
        )
        .alias(
            "retrieved_at_utc"
        )
    )
