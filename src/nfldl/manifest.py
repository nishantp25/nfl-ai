"""Typed access to ``configs/data_sources.yaml``.

The manifest is the contract. This module parses it, validates it, and exposes
it as frozen dataclasses so the rest of the codebase never string-indexes a
dict and never invents a dataset that isn't declared.

Some upstream datasets change schema at known season boundaries. Those changes
must be represented explicitly with ``schema_epochs`` rather than hidden behind
schema-drift-tolerant reads. A schema epoch may change the primary key and the
knowledge timestamp while the logical dataset name remains the same.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

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

    def resolve_end(
        self,
        current_season: int | None = None,
    ) -> int | None:
        if self.end is not None:
            return self.end

        if self.start is None:
            return None

        return (
            current_season
            if current_season is not None
            else nfl_season_of()
        )

    def seasons(
        self,
        current_season: int | None = None,
    ) -> list[int]:
        end = self.resolve_end(current_season)

        if self.start is None or end is None:
            return []

        return list(
            range(
                self.start,
                end + 1,
            )
        )

    def clip(
        self,
        wanted: Iterable[int],
        current_season: int | None = None,
    ) -> list[int]:
        available = set(
            self.seasons(current_season)
        )

        return sorted(
            season
            for season in wanted
            if season in available
        )

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.start}-"
            f"{self.end if self.end is not None else 'current'}"
        )


@dataclass(frozen=True)
class NullableKeyRule:
    """Allow one primary-key column to be null under explicit conditions."""

    column: str
    when: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class SchemaContract:
    """Schema requirements that apply to one logical slice of a dataset.

    This intentionally contains only fields whose interpretation may change
    between upstream schema generations. General dataset metadata still lives
    on :class:`DatasetSpec`.
    """

    primary_key: tuple[str, ...]
    timestamp_field: str | None
    nullable_key_rules: tuple[NullableKeyRule, ...] = ()


@dataclass(frozen=True)
class SchemaEpoch:
    """A schema contract valid for a contiguous range of NFL seasons.

    ``end is None`` means the epoch is open-ended.
    """

    start: int
    end: int | None
    primary_key: tuple[str, ...]
    timestamp_field: str | None
    nullable_key_rules: tuple[NullableKeyRule, ...] = ()

    def contains(
        self,
        season: int,
    ) -> bool:
        if season < self.start:
            return False

        if self.end is None:
            return True

        return season <= self.end

    @property
    def contract(self) -> SchemaContract:
        return SchemaContract(
            primary_key=self.primary_key,
            timestamp_field=self.timestamp_field,
            nullable_key_rules=self.nullable_key_rules,
        )


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
    schema_epochs: tuple[SchemaEpoch, ...] = ()
    nullable_key_rules: tuple[NullableKeyRule, ...] = ()
    point_in_time: str | None = None
    leakage_risk: str = "none"
    license_or_usage_notes: str = ""
    coverage_caveat: str = ""
    role: str = ""
    upstream: str = ""
    raw: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
    )

    @property
    def is_partitioned_by_season(self) -> bool:
        return "season" in self.partition_by

    @property
    def is_seasonal(self) -> bool:
        return self.declared.start is not None

    def schema_for_season(
        self,
        season: int,
    ) -> SchemaContract:
        """Return the schema contract that applies to ``season``.

        Datasets without explicit schema epochs preserve the historical
        behaviour: their top-level ``primary_key`` and ``timestamp_field`` are
        valid for every season.

        If epochs are declared, every season inside declared coverage must be
        covered by exactly one epoch. That invariant is validated while the
        manifest is loaded, so failure here means the caller requested a season
        outside the manifest's schema contract.
        """
        if not self.schema_epochs:
            return SchemaContract(
                primary_key=self.primary_key,
                timestamp_field=self.timestamp_field,
                nullable_key_rules=self.nullable_key_rules,
            )

        for epoch in self.schema_epochs:
            if epoch.contains(season):
                return epoch.contract

        raise ManifestError(
            f"dataset '{self.name}' has no schema contract for season {season}"
        )

    def storage_path(
        self,
        root: Path,
        **keys: Any,
    ) -> Path:
        """Render ``local_storage_path`` with partition keys filled in."""
        tmpl = self.local_storage_path

        missing = [
            token
            for token in (
                "{season}",
                "{stat_type}",
            )
            if (
                token in tmpl
                and token.strip("{}") not in keys
            )
        ]

        if missing:
            raise ManifestError(
                f"{self.name}: storage path needs {missing}, "
                f"got {list(keys)}"
            )

        rendered = tmpl.format(
            **keys
        )

        # Manifest paths are repo-relative and start with data/raw; re-root them.
        rel = Path(rendered)
        parts = rel.parts

        if parts[:2] == (
            "data",
            "raw",
        ):
            rel = Path(
                *parts[2:]
            )

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

    def __getitem__(
        self,
        name: str,
    ) -> DatasetSpec:
        try:
            return self.datasets[name]
        except KeyError:
            raise ManifestError(
                f"'{name}' is not declared in {self.path.name}. "
                "Add it to the manifest before writing a fetcher for it. "
                f"Known: {sorted(self.datasets)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(
            self.datasets
        )

    def required_names(self) -> list[str]:
        return sorted(
            name
            for name, spec in self.datasets.items()
            if spec.required
        )

    def tier_rank(
        self,
        tier: str,
    ) -> int:
        for entry in self.source_priority:
            if entry["tier"] == tier:
                return int(
                    entry["rank"]
                )

        raise ManifestError(
            f"unknown source tier: {tier}"
        )

    def assert_not_forbidden(
        self,
        name: str,
    ) -> None:
        if name in self.forbidden_downloads:
            raise ManifestError(
                f"'{name}' is a DERIVED feature. The manifest forbids "
                "downloading it from a third party -- compute it from "
                "canonical raw data so it can be reconstructed historically "
                "without leakage."
            )

    # -- verified coverage overlay ----------------------------------------

    def with_verified_coverage(
        self,
        report_path: Path | None = None,
    ) -> Manifest:
        """Overlay genuinely verified runtime coverage onto the manifest.

        URL or asset reachability by itself is not verification. A coverage
        report may replace the manifest's advisory coverage only when the
        verifier explicitly records ``schema_verified: true``.

        If the report is absent, incomplete, or contains only reachability
        evidence, the manifest coverage is preserved.
        """
        report_path = (
            report_path
            or COVERAGE_REPORT_PATH
        )

        if not report_path.exists():
            return self

        payload = json.loads(
            report_path.read_text(
                encoding="utf-8"
            )
        )

        found = payload.get(
            "datasets",
            {},
        )

        updated: dict[str, DatasetSpec] = {}

        for name, spec in self.datasets.items():
            probe = found.get(name)

            if not probe:
                updated[name] = spec
                continue

            if probe.get("schema_verified") is not True:
                updated[name] = spec
                continue

            verified_start = probe.get(
                "verified_start"
            )

            if verified_start is None:
                updated[name] = spec
                continue

            cov = Coverage(
                verified_start,
                probe.get(
                    "verified_end"
                ),
            )

            updated[name] = DatasetSpec(
                **{
                    **spec.__dict__,
                    "verified": cov,
                }
            )

        return Manifest(
            **{
                **self.__dict__,
                "datasets": updated,
            }
        )


def _as_tuple(
    value: Any,
) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(
        value,
        str,
    ):
        return (
            ()
            if value == "none"
            else (value,)
        )

    return tuple(
        str(item)
        for item in value
    )


def _parse_nullable_key_rules(
    value: Any,
    *,
    dataset: str,
    context: str,
    primary_key: tuple[str, ...],
) -> tuple[NullableKeyRule, ...]:
    """Parse conditional exceptions for null values in primary-key columns."""
    if value is None:
        return ()

    if not isinstance(value, list):
        raise ManifestError(
            f"dataset '{dataset}' {context}: "
            "nullable_key_rules must be a list"
        )

    rules: list[NullableKeyRule] = []

    for index, raw_rule in enumerate(value, start=1):
        if not isinstance(raw_rule, dict):
            raise ManifestError(
                f"dataset '{dataset}' {context} nullable-key rule {index} "
                "must be a mapping"
            )

        if "column" not in raw_rule or "when" not in raw_rule:
            raise ManifestError(
                f"dataset '{dataset}' {context} nullable-key rule {index} "
                "must declare both 'column' and 'when'"
            )

        column = str(raw_rule["column"]).strip()

        if not column:
            raise ManifestError(
                f"dataset '{dataset}' {context} nullable-key rule {index} "
                "has an empty column"
            )

        if column not in primary_key:
            raise ManifestError(
                f"dataset '{dataset}' {context} nullable-key rule {index} "
                f"targets '{column}', which is not in primary_key "
                f"{list(primary_key)}"
            )

        raw_when = raw_rule["when"]

        if not isinstance(raw_when, dict) or not raw_when:
            raise ManifestError(
                f"dataset '{dataset}' {context} nullable-key rule {index} "
                "'when' must be a non-empty mapping"
            )

        conditions: list[tuple[str, Any]] = []

        for raw_field, expected in raw_when.items():
            field_name = str(raw_field).strip()

            if not field_name:
                raise ManifestError(
                    f"dataset '{dataset}' {context} nullable-key rule {index} "
                    "contains an empty condition field"
                )

            if isinstance(expected, (dict, list, set, tuple)):
                raise ManifestError(
                    f"dataset '{dataset}' {context} nullable-key rule {index} "
                    f"condition '{field_name}' must compare against a scalar"
                )

            conditions.append(
                (
                    field_name,
                    expected,
                )
            )

        rules.append(
            NullableKeyRule(
                column=column,
                when=tuple(conditions),
            )
        )

    return tuple(rules)


def _parse_epoch_int(
    *,
    dataset: str,
    index: int,
    field_name: str,
    value: Any,
    allow_none: bool,
) -> int | None:
    """Parse one epoch boundary without accepting booleans or odd strings."""
    if value is None:
        if allow_none:
            return None

        raise ManifestError(
            f"dataset '{dataset}' schema epoch {index} "
            f"must declare {field_name}"
        )

    if isinstance(
        value,
        bool,
    ):
        raise ManifestError(
            f"dataset '{dataset}' schema epoch {index} "
            f"has invalid {field_name}: {value!r}"
        )

    try:
        parsed = int(value)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ManifestError(
            f"dataset '{dataset}' schema epoch {index} "
            f"has invalid {field_name}: {value!r}"
        ) from exc

    return parsed


def _parse_schema_epochs(
    *,
    name: str,
    body: dict[str, Any],
    declared: Coverage,
    required: bool,
) -> tuple[SchemaEpoch, ...]:
    """Parse and validate optional season-aware schema contracts."""
    raw_epochs = body.get(
        "schema_epochs"
    )

    if raw_epochs is None:
        return ()

    if not isinstance(
        raw_epochs,
        list,
    ):
        raise ManifestError(
            f"dataset '{name}': schema_epochs must be a list"
        )

    if not raw_epochs:
        raise ManifestError(
            f"dataset '{name}': schema_epochs may not be empty"
        )

    if declared.start is None:
        raise ManifestError(
            f"dataset '{name}': schema_epochs require seasonal coverage"
        )

    epochs: list[SchemaEpoch] = []

    for index, raw_epoch in enumerate(
        raw_epochs,
        start=1,
    ):
        if not isinstance(
            raw_epoch,
            dict,
        ):
            raise ManifestError(
                f"dataset '{name}' schema epoch {index} must be a mapping"
            )

        missing = [
            field_name
            for field_name in (
                "start",
                "end",
                "primary_key",
                "timestamp_field",
            )
            if field_name not in raw_epoch
        ]

        if missing:
            raise ManifestError(
                f"dataset '{name}' schema epoch {index} "
                f"is missing fields: {missing}"
            )

        start = _parse_epoch_int(
            dataset=name,
            index=index,
            field_name="start",
            value=raw_epoch["start"],
            allow_none=False,
        )

        end = _parse_epoch_int(
            dataset=name,
            index=index,
            field_name="end",
            value=raw_epoch["end"],
            allow_none=True,
        )

        assert start is not None

        if (
            end is not None
            and end < start
        ):
            raise ManifestError(
                f"dataset '{name}' schema epoch {index} "
                f"ends before it starts: {start}-{end}"
            )

        pk = _as_tuple(
            raw_epoch["primary_key"]
        )

        if required and not pk:
            raise ManifestError(
                f"required dataset '{name}' schema epoch {index} "
                "must declare a primary_key"
            )

        timestamp_raw = raw_epoch[
            "timestamp_field"
        ]

        timestamp_field = (
            None
            if timestamp_raw is None
            else str(timestamp_raw)
        )

        nullable_key_rules = _parse_nullable_key_rules(
            raw_epoch.get("nullable_key_rules"),
            dataset=name,
            context=f"schema epoch {index}",
            primary_key=pk,
        )

        epochs.append(
            SchemaEpoch(
                start=start,
                end=end,
                primary_key=pk,
                timestamp_field=timestamp_field,
                nullable_key_rules=nullable_key_rules,
            )
        )

    epochs.sort(
        key=lambda epoch: epoch.start
    )

    first = epochs[0]

    if first.start > declared.start:
        raise ManifestError(
            f"dataset '{name}' schema_epochs have a gap before "
            f"season {first.start}; declared coverage starts at "
            f"{declared.start}"
        )

    if first.start < declared.start:
        raise ManifestError(
            f"dataset '{name}' schema epoch starts at {first.start}, "
            f"before declared historical_start {declared.start}"
        )

    for previous, current in itertools.pairwise(epochs):
        if previous.end is None:
            raise ManifestError(
                f"dataset '{name}' schema_epochs overlap: "
                f"open-ended epoch beginning {previous.start} is followed "
                f"by another epoch beginning {current.start}"
            )

        if current.start <= previous.end:
            raise ManifestError(
                f"dataset '{name}' schema_epochs overlap at "
                f"{current.start}: previous epoch is "
                f"{previous.start}-{previous.end}"
            )

        expected_start = (
            previous.end + 1
        )

        if current.start != expected_start:
            raise ManifestError(
                f"dataset '{name}' schema_epochs have a gap between "
                f"{previous.end} and {current.start}"
            )

    last = epochs[-1]

    if declared.end is None:
        if last.end is not None:
            raise ManifestError(
                f"dataset '{name}' schema_epochs have a gap after "
                f"{last.end}; historical_end is open-ended"
            )
    else:
        if last.end is None:
            raise ManifestError(
                f"dataset '{name}' schema epoch beginning {last.start} "
                f"extends beyond declared historical_end {declared.end}"
            )

        if last.end < declared.end:
            raise ManifestError(
                f"dataset '{name}' schema_epochs have a gap after "
                f"{last.end}; declared historical_end is {declared.end}"
            )

        if last.end > declared.end:
            raise ManifestError(
                f"dataset '{name}' schema epoch ends at {last.end}, "
                f"after declared historical_end {declared.end}"
            )

    return tuple(
        epochs
    )


def _parse_dataset(
    name: str,
    body: dict[str, Any],
) -> DatasetSpec:
    missing = [
        field_name
        for field_name in REQUIRED_FIELDS
        if field_name not in body
    ]

    if missing:
        raise ManifestError(
            f"dataset '{name}' is missing manifest fields: {missing}"
        )

    tier = body.get(
        "tier",
        "reputable_public",
    )

    if tier not in VALID_TIERS:
        raise ManifestError(
            f"dataset '{name}' has unknown tier '{tier}'"
        )

    declared = Coverage(
        body["historical_start"],
        body["historical_end"],
    )

    verified = Coverage(
        body.get(
            "verified_start",
            declared.start,
        ),
        body.get(
            "verified_end",
            declared.end,
        ),
    )

    if (
        declared.start is not None
        and verified.start is not None
        and verified.start < declared.start
    ):
        raise ManifestError(
            f"dataset '{name}': verified_start {verified.start} "
            f"precedes declared historical_start {declared.start}"
        )

    pk = _as_tuple(
        body["primary_key"]
    )

    required = bool(
        body["required"]
    )

    if not pk and required:
        raise ManifestError(
            f"required dataset '{name}' must declare a primary_key"
        )

    timestamp_raw = body[
        "timestamp_field"
    ]

    timestamp_field = (
        None
        if timestamp_raw is None
        else str(timestamp_raw)
    )

    nullable_key_rules = _parse_nullable_key_rules(
        body.get("nullable_key_rules"),
        dataset=name,
        context="top-level schema",
        primary_key=pk,
    )

    schema_epochs = _parse_schema_epochs(
        name=name,
        body=body,
        declared=declared,
        required=required,
    )

    return DatasetSpec(
        name=name,
        provider=str(
            body["provider"]
        ),
        tier=tier,
        access=str(
            body["access"]
        ),
        declared=declared,
        verified=verified,
        update_frequency=str(
            body["update_frequency"]
        ),
        local_storage_path=str(
            body["local_storage_path"]
        ),
        primary_key=pk,
        timestamp_field=timestamp_field,
        partition_by=_as_tuple(
            body.get(
                "partition_by",
                "none",
            )
        ),
        required=required,
        variants=_as_tuple(
            body.get(
                "variants"
            )
        ),
        schema_epochs=schema_epochs,
        nullable_key_rules=nullable_key_rules,
        point_in_time=body.get(
            "point_in_time"
        ),
        leakage_risk=str(
            body.get(
                "leakage_risk",
                "none",
            )
        ),
        license_or_usage_notes=str(
            body.get(
                "license_or_usage_notes",
                "",
            )
        ),
        coverage_caveat=str(
            body.get(
                "coverage_caveat",
                "",
            )
        ),
        role=str(
            body.get(
                "role",
                "",
            )
        ),
        upstream=str(
            body.get(
                "upstream",
                "",
            )
        ),
        raw=body,
    )


def load_manifest(
    path: Path | None = None,
) -> Manifest:
    path = (
        path
        or MANIFEST_PATH
    )

    if not path.exists():
        raise ManifestError(
            f"source manifest not found at {path}"
        )

    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ManifestError(
            f"source manifest at {path} must contain a YAML mapping"
        )

    if "datasets" not in payload:
        raise ManifestError(
            f"source manifest at {path} has no datasets section"
        )

    meta = payload.get(
        "manifest",
        {},
    )

    datasets = {
        name: _parse_dataset(
            name,
            body,
        )
        for name, body in payload[
            "datasets"
        ].items()
    }

    priority = tuple(
        payload.get(
            "source_priority",
            (),
        )
    )

    ranks = [
        entry["rank"]
        for entry in priority
    ]

    if sorted(ranks) != list(
        range(
            1,
            len(ranks) + 1,
        )
    ):
        raise ManifestError(
            "source_priority ranks must be 1..N with no gaps, "
            f"got {ranks}"
        )

    forbidden = frozenset(
        payload.get(
            "forbidden_downloads",
            (),
        )
    )

    overlap = (
        forbidden
        & set(datasets)
    )

    if overlap:
        raise ManifestError(
            "these names are declared as datasets AND listed as forbidden "
            f"derived features: {sorted(overlap)}"
        )

    # Every seasonal dataset must partition by season, or storage will collide.
    for name, spec in datasets.items():
        if (
            "{season}" in spec.local_storage_path
            and not spec.is_partitioned_by_season
        ):
            raise ManifestError(
                f"dataset '{name}' templates {{season}} into its path "
                "but does not declare partition_by: season"
            )

    return Manifest(
        version=int(
            meta.get(
                "version",
                1,
            )
        ),
        verified_on=str(
            meta.get(
                "verified_on",
                "unverified",
            )
        ),
        defaults=payload.get(
            "defaults",
            {},
        ),
        datasets=datasets,
        source_priority=priority,
        forbidden_downloads=forbidden,
        path=path,
    )


@lru_cache(maxsize=4)
def get_manifest(
    path: str | None = None,
) -> Manifest:
    return load_manifest(
        Path(path)
        if path
        else None
    ).with_verified_coverage()