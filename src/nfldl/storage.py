"""The raw data lake: atomic parquet writes with a provenance sidecar per file.

Every partition written here answers, on its own, "where did this come from and
when did we learn it". That is the whole point of the source manifest, and it is
what makes historical reconstruction auditable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .config import PROVENANCE_LOG_PATH, RAW_DIR
from .manifest import DatasetSpec

log = logging.getLogger(__name__)

# Columns this layer adds to every raw table. Downstream code must exclude them
# from feature sets -- `_ingested_at` in particular is a *processing* time and
# says nothing about when the fact became knowable.
INGEST_COLUMNS = ("_ingested_at", "_source_tier", "_provider")


@dataclass(frozen=True)
class PartitionRecord:
    """Provenance for one written partition."""

    dataset: str
    path: str
    partition: dict[str, Any]
    provider: str
    tier: str
    access: str
    rows: int
    columns: int
    column_names: list[str]
    content_sha256: str
    ingested_at: str
    manifest_version: int
    manifest_verified_on: str
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_parquet(df: pl.DataFrame, dest: Path) -> None:
    """Write via a temp file in the destination directory, then rename.

    A rename on the same filesystem is atomic, so an interrupted ingest can
    never leave a half-written partition that a later run mistakes for complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".parquet.tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.write_parquet(tmp, compression="zstd", statistics=True)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def stamp(df: pl.DataFrame, spec: DatasetSpec, ingested_at: datetime) -> pl.DataFrame:
    """Attach ingest provenance columns, overwriting any that already exist."""
    return df.with_columns(
        pl.lit(ingested_at).cast(pl.Datetime("us", "UTC")).alias("_ingested_at"),
        pl.lit(spec.tier).alias("_source_tier"),
        pl.lit(spec.provider).alias("_provider"),
    )


def write_partition(
    df: pl.DataFrame,
    spec: DatasetSpec,
    *,
    root: Path | None = None,
    notes: str = "",
    **partition_keys: Any,
) -> PartitionRecord:
    """Write one partition and record its provenance.

    Returns the :class:`PartitionRecord`, which is also appended to
    ``reports/ingestion_log.jsonl`` and written next to the parquet file.
    """
    from .manifest import get_manifest  # local import avoids a cycle

    root = root or RAW_DIR
    now = datetime.now(timezone.utc)
    dest = spec.storage_path(root, **partition_keys)
    stamped = stamp(df, spec, now)

    _atomic_write_parquet(stamped, dest)

    mf = get_manifest()
    record = PartitionRecord(
        dataset=spec.name,
        path=str(dest),
        partition=partition_keys,
        provider=spec.provider,
        tier=spec.tier,
        access=spec.access,
        rows=stamped.height,
        columns=stamped.width,
        column_names=stamped.columns,
        content_sha256=_sha256_of(dest),
        ingested_at=now.isoformat(),
        manifest_version=mf.version,
        manifest_verified_on=mf.verified_on,
        notes=notes,
    )

    dest.with_suffix(".provenance.json").write_text(
        json.dumps(asdict(record), indent=2)
    )
    PROVENANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROVENANCE_LOG_PATH.open("a") as fh:
        fh.write(record.to_json() + "\n")

    log.info("wrote %s %s -> %s rows", spec.name, partition_keys or "", f"{stamped.height:,}")
    return record


def partition_exists(spec: DatasetSpec, *, root: Path | None = None, **keys: Any) -> bool:
    return spec.storage_path(root or RAW_DIR, **keys).exists()


def read_dataset(
    spec: DatasetSpec,
    *,
    root: Path | None = None,
    seasons: list[int] | None = None,
    stat_type: str | None = None,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Read a dataset back out of the lake.

    Season-partitioned datasets are concatenated with ``diagonal_relaxed`` so
    that genuine schema drift across seasons (nflverse does this regularly)
    widens the frame instead of raising. Columns absent in a season stay null,
    which is the honest representation: the field did not exist that year.
    """
    root = root or RAW_DIR
    paths: list[Path] = []

    if spec.is_partitioned_by_season:
        wanted = seasons if seasons is not None else spec.verified.seasons()
        for season in wanted:
            keys: dict[str, Any] = {"season": season}
            if "{stat_type}" in spec.local_storage_path:
                if stat_type is None:
                    raise ValueError(f"{spec.name} requires stat_type")
                keys["stat_type"] = stat_type
            p = spec.storage_path(root, **keys)
            if p.exists():
                paths.append(p)
    else:
        keys = {}
        if "{stat_type}" in spec.local_storage_path:
            if stat_type is None:
                raise ValueError(f"{spec.name} requires stat_type")
            keys["stat_type"] = stat_type
        p = spec.storage_path(root, **keys)
        if p.exists():
            paths.append(p)

    if not paths:
        raise FileNotFoundError(
            f"no local partitions for '{spec.name}'. Run: nfldl ingest {spec.name}"
        )

    frames = [pl.scan_parquet(p) for p in paths]
    out = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
    return out if lazy else out.collect()


def inventory(root: Path | None = None) -> pl.DataFrame:
    """Everything currently in the lake, from the provenance sidecars."""
    root = root or RAW_DIR
    rows = [
        json.loads(p.read_text())
        for p in sorted(root.rglob("*.provenance.json"))
    ]
    if not rows:
        return pl.DataFrame(
            schema={"dataset": pl.String, "path": pl.String, "rows": pl.Int64}
        )
    for r in rows:
        r["partition"] = json.dumps(r.get("partition", {}))
        r["column_names"] = len(r.get("column_names", []))
    return pl.DataFrame(rows).sort(["dataset", "path"])
