"""Typed access to ``configs/data_sources.yaml``.

The manifest is the contract. This module parses it, validates it, and exposes
it as frozen dataclasses so the rest of the codebase never string-indexes a
dict and never invents a dataset that isn't declared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import COVERAGE_REPORT_PATH, MANIFEST_PATH, nfl_season_of

REQUIRED_FIELDS = (
    "provider",
    "access",
    "historical_start",
    "historical_end",
    "update_frequency",
    "local_storage_path",
    "primary_key",
    "timestamp_field",
    "required",
)

VALID_TIERS = {
    "official_nfl",
    "nflverse",
    "established_db",
    "reputable_public",
    "curated_fallback",
}


class ManifestError(ValueError):
    """Raised when the manifest is internally inconsistent."""


@dataclass(frozen=True)
class Coverage:
    """A season range. ``end is None`` means 'through the current season'."""

    start: int | None
    end: int | None

    def resolve_end(self, current_season: int | None = None) -> int | None:
        if self.end is not None:
            return self.end
        if self.start is None:
            return None
        return current_season if current_season is not None else nfl_season_of()

    def seasons(self, current_season: int | None = None) -> list[int]:
        end = self.resolve_end(current_season)
        if self.start is None or end is None:
            return []
        return list(range(self.start, end + 1))

    def clip(self, wanted: Iterable[int], current_season: int | None = None) -> list[int]:
        available = set(self.seasons(current_season))
        return sorted(s for s in wanted if s in available)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.start}-{self.end if self.end is not None else 'current'}"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    provider: str
    tier: str
    access: str
    declared: Coverage
    verified: Coverage
    update_frequency: str
    local_storage_path: str
    primary_key: tuple[str, ...]
    timestamp_field: str | None
    partition_by: tuple[str, ...]
    required: bool
    variants: tuple[str, ...] = ()
    point_in_time: str | None = None
    leakage_risk: str = "none"
    license_or_usage_notes: str = ""
    coverage_caveat: str = ""
    role: str = ""
    upstream: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_partitioned_by_season(self) -> bool:
        return "season" in self.partition_by

    @property
    def is_seasonal(self) -> bool:
        return self.declared.start is not None

    def storage_path(self, root: Path, **keys: Any) -> Path:
        """Render ``local_storage_path`` with partition keys filled in."""
        tmpl = self.local_storage_path
        missing = [
            token
            for token in ("{season}", "{stat_type}")
            if token in tmpl and token.strip("{}") not in keys
        ]
        if missing:
            raise ManifestError(f"{self.name}: storage path needs {missing}, got {list(keys)}")
        rendered = tmpl.format(**keys)
        # Manifest paths are repo-relative and start with data/raw; re-root them.
        rel = Path(rendered)
        parts = rel.parts
        if parts[:2] == ("data", "raw"):
            rel = Path(*parts[2:])
        return root / rel


@dataclass(frozen=True)
class Manifest:
    version: int
    verified_on: str
    defaults: dict[str, Any]
    datasets: dict[str, DatasetSpec]
    source_priority: tuple[dict[str, Any], ...]
    forbidden_downloads: frozenset[str]
    path: Path

    # -- lookups -----------------------------------------------------------
    def __getitem__(self, name: str) -> DatasetSpec:
        try:
            return self.datasets[name]
        except KeyError:
            raise ManifestError(
                f"'{name}' is not declared in {self.path.name}. Add it to the manifest "
                f"before writing a fetcher for it. Known: {sorted(self.datasets)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self.datasets)

    def required_names(self) -> list[str]:
        return sorted(n for n, s in self.datasets.items() if s.required)

    def tier_rank(self, tier: str) -> int:
        for entry in self.source_priority:
            if entry["tier"] == tier:
                return int(entry["rank"])
        raise ManifestError(f"unknown source tier: {tier}")

    def assert_not_forbidden(self, name: str) -> None:
        if name in self.forbidden_downloads:
            raise ManifestError(
                f"'{name}' is a DERIVED feature. The manifest forbids downloading it from a "
                "third party -- compute it from canonical raw data so it can be reconstructed "
                "historically without leakage."
            )

    # -- verified coverage overlay ----------------------------------------
    def with_verified_coverage(self, report_path: Path | None = None) -> "Manifest":
        """Overlay coverage discovered by ``nfldl verify`` onto the manifest.

        The report is advisory: if it is absent or stale we fall back to the
        declared values, but callers can tell the difference.
        """
        report_path = report_path or COVERAGE_REPORT_PATH
        if not report_path.exists():
            return self
        payload = json.loads(report_path.read_text())
        found = payload.get("datasets", {})
        updated: dict[str, DatasetSpec] = {}
        for name, spec in self.datasets.items():
            probe = found.get(name)
            if not probe or probe.get("verified_start") is None:
                updated[name] = spec
                continue
            cov = Coverage(probe["verified_start"], probe["verified_end"])
            updated[name] = DatasetSpec(**{**spec.__dict__, "verified": cov})
        return Manifest(**{**self.__dict__, "datasets": updated})


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return () if value == "none" else (value,)
    return tuple(str(v) for v in value)


def _parse_dataset(name: str, body: dict[str, Any]) -> DatasetSpec:
    missing = [f for f in REQUIRED_FIELDS if f not in body]
    if missing:
        raise ManifestError(f"dataset '{name}' is missing manifest fields: {missing}")

    tier = body.get("tier", "reputable_public")
    if tier not in VALID_TIERS:
        raise ManifestError(f"dataset '{name}' has unknown tier '{tier}'")

    declared = Coverage(body["historical_start"], body["historical_end"])
    verified = Coverage(
        body.get("verified_start", declared.start),
        body.get("verified_end", declared.end),
    )
    if declared.start is not None and verified.start is not None:
        if verified.start < declared.start:
            raise ManifestError(
                f"dataset '{name}': verified_start {verified.start} precedes declared "
                f"historical_start {declared.start}"
            )

    pk = _as_tuple(body["primary_key"])
    if not pk and body["required"]:
        raise ManifestError(f"required dataset '{name}' must declare a primary_key")

    return DatasetSpec(
        name=name,
        provider=str(body["provider"]),
        tier=tier,
        access=str(body["access"]),
        declared=declared,
        verified=verified,
        update_frequency=str(body["update_frequency"]),
        local_storage_path=str(body["local_storage_path"]),
        primary_key=pk,
        timestamp_field=body["timestamp_field"],
        partition_by=_as_tuple(body.get("partition_by", "none")),
        required=bool(body["required"]),
        variants=_as_tuple(body.get("variants")),
        point_in_time=body.get("point_in_time"),
        leakage_risk=str(body.get("leakage_risk", "none")),
        license_or_usage_notes=str(body.get("license_or_usage_notes", "")),
        coverage_caveat=str(body.get("coverage_caveat", "")),
        role=str(body.get("role", "")),
        upstream=str(body.get("upstream", "")),
        raw=body,
    )


def load_manifest(path: Path | None = None) -> Manifest:
    path = path or MANIFEST_PATH
    if not path.exists():
        raise ManifestError(f"source manifest not found at {path}")
    payload = yaml.safe_load(path.read_text())

    meta = payload.get("manifest", {})
    datasets = {
        name: _parse_dataset(name, body) for name, body in payload["datasets"].items()
    }

    priority = tuple(payload.get("source_priority", ()))
    ranks = [p["rank"] for p in priority]
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ManifestError(f"source_priority ranks must be 1..N with no gaps, got {ranks}")

    forbidden = frozenset(payload.get("forbidden_downloads", ()))
    overlap = forbidden & set(datasets)
    if overlap:
        raise ManifestError(
            f"these names are declared as datasets AND listed as forbidden derived "
            f"features: {sorted(overlap)}"
        )

    # Every seasonal dataset must partition by season, or storage will collide.
    for name, spec in datasets.items():
        if "{season}" in spec.local_storage_path and not spec.is_partitioned_by_season:
            raise ManifestError(
                f"dataset '{name}' templates {{season}} into its path but does not declare "
                "partition_by: season"
            )

    return Manifest(
        version=int(meta.get("version", 1)),
        verified_on=str(meta.get("verified_on", "unverified")),
        defaults=payload.get("defaults", {}),
        datasets=datasets,
        source_priority=priority,
        forbidden_downloads=forbidden,
        path=path,
    )


@lru_cache(maxsize=4)
def get_manifest(path: str | None = None) -> Manifest:
    return load_manifest(Path(path) if path else None).with_verified_coverage()
