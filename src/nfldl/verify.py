"""Verify what actually exists before ingesting anything.

The project must never confuse four different kinds of evidence:

1. Declared coverage:
   what the manifest/provider documentation says should exist.

2. Asset reachability:
   whether a remote release asset responds successfully to a HEAD request.

3. Observed coverage:
   seasons actually seen inside data that we downloaded and inspected.

4. Verified coverage:
   coverage supported by downloaded content whose schema contract passed.

A successful HEAD request proves only that an object is reachable. It does not
prove the object's schema, internal season range, primary-key validity, or
point-in-time timestamp contract.

For seasonal datasets, ``nfldl verify`` can cheaply HEAD-probe all candidate
season assets. ``nfldl verify --sample`` additionally downloads a representative
season and validates its real schema, but one sampled season does not certify
the schema of every historical season.

For a single all-history file, downloading the complete object may establish
its internal observed range when the adapter actually returns the whole object.

Output goes to ``reports/verified_coverage.json``. Only content-backed ranges
with ``schema_verified: true`` may override manifest coverage or be written back
to ``configs/data_sources.yaml``.
"""

from __future__ import annotations

import inspect
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from requests.exceptions import RequestException

from .config import (
    COVERAGE_REPORT_PATH,
    MANIFEST_PATH,
    get_settings,
    nfl_season_of,
)
from .http import head_ok
from .manifest import DatasetSpec, get_manifest
from .registry import REGISTRY, get_adapter

log = logging.getLogger(__name__)

PROBE_MARGIN_BACK = 3
PROBE_MARGIN_FORWARD = 1
PROBE_FLOOR = 1999

FETCH_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
    KeyError,
    RequestException,
    pl.exceptions.PolarsError,
)


@dataclass
class DatasetProbe:
    """Evidence collected for one dataset."""

    dataset: str
    provider: str

    declared_start: int | None
    declared_end: int | None

    asset_reachable: bool | None = None
    asset_start: int | None = None
    asset_end: int | None = None

    observed_start: int | None = None
    observed_end: int | None = None

    schema_verified: bool = False

    verified_start: int | None = None
    verified_end: int | None = None

    seasons_present: list[int] = field(default_factory=list)

    seasons_missing_in_declared_range: list[int] = field(
        default_factory=list
    )

    target_window_gap: list[int] = field(
        default_factory=list
    )

    probe_method: str = "head"

    sampled_season: int | None = None
    sampled_rows: int | None = None
    sampled_columns: list[str] = field(default_factory=list)
    sampled_variants: list[str] = field(default_factory=list)

    notes: list[str] = field(default_factory=list)

    ok: bool = True


def _variants_for(
    spec: DatasetSpec,
    adapter: Any,
) -> list[str | None]:
    """Return every declared asset variant that should be inspected."""
    if not adapter.variant_assets:
        return [None]

    if spec.variants:
        return list(spec.variants)

    return [None]


def _probe_seasons(
    spec: DatasetSpec,
    current_season: int,
) -> tuple[list[int], int]:
    """Return HEAD-reachable seasons and the earliest season actually probed."""
    adapter = get_adapter(spec.name)

    if adapter.asset_url is None:
        return [], current_season

    settings = get_settings()

    target_start = getattr(
        settings,
        "target_season_start",
        settings.target_seasons[0],
    )

    floor = max(
        PROBE_FLOOR,
        min(
            target_start,
            spec.declared.start or PROBE_FLOOR,
        ),
    )

    start = max(
        (spec.declared.start or current_season) - PROBE_MARGIN_BACK,
        floor,
    )

    end = current_season + PROBE_MARGIN_FORWARD

    candidates = list(
        range(
            start,
            end + 1,
        )
    )

    variants = _variants_for(
        spec,
        adapter,
    )

    def present(
        season: int,
    ) -> int | None:
        for variant in variants:
            keys: dict[str, Any] = {
                "season": season,
            }

            if variant is not None:
                keys["stat_type"] = variant

            url = adapter.asset_url(
                **keys
            )

            if url and head_ok(url):
                return season

        return None

    with ThreadPoolExecutor(
        max_workers=settings.max_workers
    ) as pool:
        found = list(
            pool.map(
                present,
                candidates,
            )
        )

    return (
        sorted(
            season
            for season in found
            if season is not None
        ),
        candidates[0],
    )


def _extract_seasons(
    df: pl.DataFrame,
) -> list[int]:
    """Extract a usable historical-season field from downloaded content."""
    season_col = next(
        (
            column
            for column in (
                "season",
                "year_signed",
                "draft_year",
            )
            if column in df.columns
        ),
        None,
    )

    if season_col is None:
        return []

    seasons: set[int] = set()

    for value in (
        df.get_column(
            season_col
        )
        .drop_nulls()
        .to_list()
    ):
        try:
            seasons.add(
                int(value)
            )
        except (TypeError, ValueError):
            continue

    return sorted(
        seasons
    )


def _merge_columns(
    existing: list[str],
    new_columns: list[str],
) -> list[str]:
    """Merge column names while preserving first-seen order."""
    seen = set(existing)
    merged = list(existing)

    for column in new_columns:
        if column not in seen:
            seen.add(column)
            merged.append(column)

    return merged


def _set_observed_range(
    probe: DatasetProbe,
    seasons: list[int],
) -> None:
    """Expand a probe's observed range from downloaded content."""
    if not seasons:
        return

    low = min(seasons)
    high = max(seasons)

    if probe.observed_start is None:
        probe.observed_start = low
    else:
        probe.observed_start = min(
            probe.observed_start,
            low,
        )

    if probe.observed_end is None:
        probe.observed_end = high
    else:
        probe.observed_end = max(
            probe.observed_end,
            high,
        )


def _timestamp_field(
    spec: DatasetSpec,
    season: int | None = None,
) -> str | None:
    """Normalize the timestamp declaration active for a sampled season."""
    value = (
        spec.schema_for_season(season).timestamp_field
        if season is not None
        else spec.timestamp_field
    )

    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {
        "none",
        "null",
    }:
        return None

    return text


def _check_primary_key(
    df: pl.DataFrame,
    spec: DatasetSpec,
    probe: DatasetProbe,
    *,
    season: int | None = None,
) -> bool:
    """Validate that the active primary key exists and is unique."""
    primary_key = (
        spec.schema_for_season(season).primary_key
        if season is not None
        else spec.primary_key
    )

    absent = [
        column
        for column in primary_key
        if column not in df.columns
    ]

    if absent:
        probe.ok = False

        probe.notes.append(
            "declared primary_key columns absent from the real schema: "
            f"{absent}. Fix the manifest -- a wrong key silently permits "
            "duplicates."
        )

        return False

    if not primary_key:
        return True

    pk = list(primary_key)

    duplicate_rows = (
        df.height
        - df.select(
            pk
        )
        .unique()
        .height
    )

    if duplicate_rows:
        probe.ok = False

        probe.notes.append(
            f"primary key {pk} is not unique: "
            f"{duplicate_rows} duplicate rows"
        )

        return False

    return True


def _check_schema(
    df: pl.DataFrame,
    spec: DatasetSpec,
    probe: DatasetProbe,
    *,
    season: int | None = None,
) -> bool:
    """Validate the minimum schema contract needed to trust a sample."""
    valid = True

    if df.is_empty():
        probe.ok = False
        valid = False

        probe.notes.append(
            "sample fetch returned zero rows; schema coverage cannot be "
            "verified from an empty sample"
        )

    if not _check_primary_key(
        df,
        spec,
        probe,
        season=season,
    ):
        valid = False

    timestamp = _timestamp_field(
        spec,
        season,
    )

    if (
        timestamp is not None
        and timestamp not in df.columns
    ):
        probe.ok = False
        valid = False

        probe.notes.append(
            f"declared timestamp field '{timestamp}' is absent from the "
            "real schema"
        )

    return valid


def _single_file_requires_season(
    fetcher: Any,
) -> bool:
    """Return whether a nominal all-history adapter requires a season input."""
    try:
        signature = inspect.signature(fetcher)
    except (TypeError, ValueError):
        return False

    parameter = signature.parameters.get(
        "season"
    )

    if parameter is None:
        return False

    if parameter.default is not inspect.Parameter.empty:
        return False

    return parameter.kind in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def _single_file_sample_season(
    spec: DatasetSpec,
    current_season: int,
) -> int:
    """Pick one sensible season when an adapter requires a season argument."""
    declared_end = (
        spec.declared.end
        if spec.declared.end is not None
        else current_season
    )

    candidate = min(
        current_season,
        declared_end,
    )

    if (
        spec.declared.start is not None
        and candidate < spec.declared.start
    ):
        candidate = spec.declared.start

    return candidate


def _sample_seasonal(
    spec: DatasetSpec,
    probe: DatasetProbe,
    season: int,
) -> None:
    """Download and inspect one season without certifying the whole range."""
    adapter = get_adapter(
        spec.name
    )

    variants = _variants_for(
        spec,
        adapter,
    )

    fetched_any = False
    all_valid = True
    observed: set[int] = set()
    total_rows = 0
    columns: list[str] = []

    probe.sampled_season = season

    for variant in variants:
        kwargs: dict[str, Any] = {
            "season": season,
        }

        if variant is not None:
            kwargs["stat_type"] = variant

        label = (
            f" variant={variant}"
            if variant is not None
            else ""
        )

        try:
            df = adapter.fetch(
                **kwargs
            )
        except FETCH_EXCEPTIONS as exc:
            probe.ok = False
            all_valid = False

            probe.notes.append(
                f"sample fetch for {season}{label} failed: {exc}"
            )

            continue

        fetched_any = True

        if variant is not None:
            probe.sampled_variants.append(
                variant
            )

        total_rows += df.height

        columns = _merge_columns(
            columns,
            list(df.columns),
        )

        sample_seasons = _extract_seasons(
            df
        )

        if sample_seasons:
            observed.update(
                sample_seasons
            )
        elif df.height:
            observed.add(
                season
            )

        if (
            sample_seasons
            and season not in sample_seasons
        ):
            probe.ok = False
            all_valid = False

            probe.notes.append(
                f"sample fetch requested season {season}{label} but returned "
                f"season values {sample_seasons}"
            )

        if not _check_schema(
            df,
            spec,
            probe,
            season=season,
        ):
            all_valid = False

    if fetched_any:
        probe.sampled_rows = total_rows
        probe.sampled_columns = columns

    observed_list = sorted(
        observed
    )

    _set_observed_range(
        probe,
        observed_list,
    )

    probe.schema_verified = (
        fetched_any
        and all_valid
    )

    if probe.schema_verified:
        probe.probe_method = (
            "head+sampled-season"
        )

        probe.notes.append(
            f"schema verified from downloaded season {season}; "
            "the full HEAD-visible historical range is not schema-verified "
            "from a single season sample"
        )

    # A single sampled seasonal asset cannot certify the schema of the entire
    # historical range, so verified_start / verified_end remain unset.


def _sample_single_file(
    spec: DatasetSpec,
    probe: DatasetProbe,
    current_season: int,
) -> None:
    """Inspect a single remote object or a season-filtered adapter view."""
    adapter = get_adapter(
        spec.name
    )

    requires_season = _single_file_requires_season(
        adapter.fetch
    )

    requested_season = (
        _single_file_sample_season(
            spec,
            current_season,
        )
        if requires_season
        else None
    )

    variants = _variants_for(
        spec,
        adapter,
    )

    fetched_any = False
    all_valid = True
    total_rows = 0
    columns: list[str] = []
    observed_union: set[int] = set()
    observed_ranges: list[tuple[int, int]] = []

    for variant in variants:
        kwargs: dict[str, Any] = {}

        if requested_season is not None:
            kwargs["season"] = requested_season

        if variant is not None:
            kwargs["stat_type"] = variant

        label_parts: list[str] = []

        if requested_season is not None:
            label_parts.append(
                f"season={requested_season}"
            )

        if variant is not None:
            label_parts.append(
                f"variant={variant}"
            )

        label = (
            " " + " ".join(label_parts)
            if label_parts
            else ""
        )

        try:
            df = adapter.fetch(
                **kwargs
            )
        except FETCH_EXCEPTIONS as exc:
            probe.ok = False
            all_valid = False

            probe.notes.append(
                f"sample fetch{label} failed: {exc}"
            )

            continue

        fetched_any = True

        if variant is not None:
            probe.sampled_variants.append(
                variant
            )

        total_rows += df.height

        columns = _merge_columns(
            columns,
            list(df.columns),
        )

        seasons = _extract_seasons(
            df
        )

        if seasons:
            observed_union.update(
                seasons
            )

            observed_ranges.append(
                (
                    seasons[0],
                    seasons[-1],
                )
            )

        elif (
            requested_season is not None
            and df.height
        ):
            observed_union.add(
                requested_season
            )

            observed_ranges.append(
                (
                    requested_season,
                    requested_season,
                )
            )

        if not _check_schema(
            df,
            spec,
            probe,
            season=requested_season,
        ):
            all_valid = False

    if fetched_any:
        probe.sampled_rows = total_rows
        probe.sampled_columns = columns

    if requested_season is not None:
        probe.sampled_season = requested_season

    observed = sorted(
        observed_union
    )

    if observed:
        probe.seasons_present = observed

        _set_observed_range(
            probe,
            observed,
        )

    probe.schema_verified = (
        fetched_any
        and all_valid
    )

    if fetched_any:
        probe.probe_method = (
            "sampled-single-file"
        )

    complete_object_seen = not requires_season

    ranges_agree = (
        bool(observed_ranges)
        and len(set(observed_ranges)) == 1
    )

    if (
        probe.schema_verified
        and complete_object_seen
        and ranges_agree
    ):
        probe.verified_start = (
            observed_ranges[0][0]
        )

        probe.verified_end = (
            observed_ranges[0][1]
        )

    elif (
        probe.schema_verified
        and complete_object_seen
        and len(observed_ranges) > 1
        and not ranges_agree
    ):
        probe.notes.append(
            "sampled variants expose different internal season ranges; "
            "dataset-level verified coverage was not collapsed into one range"
        )

    elif (
        probe.schema_verified
        and requires_season
    ):
        probe.notes.append(
            "adapter requires a season argument, so this sample verifies only "
            "the returned season view and does not establish the internal "
            "coverage of the underlying all-history asset"
        )


def _record_seasonal_gaps(
    spec: DatasetSpec,
    probe: DatasetProbe,
    present: list[int],
    start_probed: int,
) -> None:
    """Record HEAD-visible seasonal gaps without calling them verified."""
    if not present:
        return

    contiguous = set(
        range(
            present[0],
            present[-1] + 1,
        )
    )

    holes = sorted(
        contiguous
        - set(present)
    )

    if holes:
        probe.seasons_missing_in_declared_range = (
            holes
        )

        probe.notes.append(
            f"non-contiguous reachable seasonal assets; missing {holes}"
        )

    declared_start = (
        spec.declared.start
    )

    if (
        declared_start is not None
        and present[0] > declared_start
    ):
        if present[0] > start_probed:
            probe.ok = False

            probe.notes.append(
                f"manifest claims data from {declared_start}, but no seasonal "
                f"asset was reachable before {present[0]}"
            )
        else:
            probe.notes.append(
                f"declared start is {declared_start}; probing began at "
                f"{start_probed}, so earlier seasons were not checked"
            )


def _record_target_gap(
    probe: DatasetProbe,
    available: list[int],
) -> None:
    """Record project-window seasons with no evidence of availability."""
    wanted = set(
        get_settings().target_seasons
    )

    gap = sorted(
        wanted
        - set(available)
    )

    if not gap:
        return

    probe.target_window_gap = gap

    probe.notes.append(
        f"{len(gap)} of the {len(wanted)} target seasons have no "
        f"availability evidence: {gap}"
    )


def probe_dataset(
    name: str,
    *,
    sample: bool = False,
) -> DatasetProbe:
    """Collect reachability, content, schema, and coverage evidence."""
    mf = get_manifest()

    spec = mf[
        name
    ]

    adapter = get_adapter(
        name
    )

    current = nfl_season_of()

    probe = DatasetProbe(
        dataset=name,
        provider=spec.provider,
        declared_start=spec.declared.start,
        declared_end=spec.declared.end,
    )

    if adapter.asset_url is None:
        probe.probe_method = (
            "derived"
        )

        probe.asset_reachable = None

        probe.notes.append(
            f"built locally from "
            f"{list(adapter.needs_local) or 'curated config'}; "
            "coverage and schema must be validated from built output after "
            "its dependencies are ingested"
        )

        return probe

    if not adapter.seasonal_assets:
        keys: dict[str, Any] = {}

        if (
            adapter.variant_assets
            and spec.variants
        ):
            keys[
                "stat_type"
            ] = spec.variants[0]

        url = adapter.asset_url(
            **keys
        )

        probe.probe_method = (
            "head-single-file"
        )

        probe.asset_reachable = bool(
            url
            and head_ok(
                url
            )
        )

        if not probe.asset_reachable:
            probe.ok = False

            probe.notes.append(
                f"asset not reachable: {url}"
            )

            return probe

        if not sample:
            probe.notes.append(
                "single remote asset is reachable, but HEAD cannot prove its "
                "internal season range or schema; use `nfldl verify --sample` "
                "for content-backed evidence"
            )

            return probe

        _sample_single_file(
            spec,
            probe,
            current,
        )

        if (
            probe.verified_start is not None
            and probe.seasons_present
        ):
            _record_target_gap(
                probe,
                probe.seasons_present,
            )

        return probe

    present, start_probed = (
        _probe_seasons(
            spec,
            current,
        )
    )

    probe.seasons_present = (
        present
    )

    probe.asset_reachable = bool(
        present
    )

    if not present:
        probe.ok = False

        probe.notes.append(
            "no seasonal assets found -- the release tag, filename pattern, "
            "or declared coverage may have changed"
        )

        return probe

    probe.asset_start = (
        present[0]
    )

    probe.asset_end = (
        present[-1]
    )

    _record_seasonal_gaps(
        spec,
        probe,
        present,
        start_probed,
    )

    _record_target_gap(
        probe,
        present,
    )

    if sample:
        _sample_seasonal(
            spec,
            probe,
            present[-1],
        )
    else:
        probe.notes.append(
            "seasonal coverage shown by HEAD is reachability evidence only; "
            "no schema or historical content range was certified"
        )

    return probe


def verify_all(
    names: list[str] | None = None,
    *,
    sample: bool = False,
) -> dict[str, Any]:
    """Verify selected datasets and write a machine-readable report."""
    mf = get_manifest()

    names = (
        names
        or sorted(
            REGISTRY
        )
    )

    probes: dict[
        str,
        DatasetProbe,
    ] = {}

    for name in names:
        log.info(
            "probing %s",
            name,
        )

        probes[
            name
        ] = probe_dataset(
            name,
            sample=sample,
        )

    nflreadpy_season = None

    try:
        import nflreadpy as nfl

        nflreadpy_season = (
            nfl.get_current_season()
        )

    except (
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        log.debug(
            "unable to read nflreadpy current-season heuristic: %s",
            exc,
        )

    ours = nfl_season_of()

    warnings: list[str] = []

    if (
        nflreadpy_season is not None
        and nflreadpy_season != ours
    ):
        warnings.append(
            f"season heuristics disagree: nfldl says {ours}, nflreadpy says "
            f"{nflreadpy_season}. The direct-asset fallback may be required "
            "near the season rollover."
        )

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "sample_requested": sample,
        "current_season_nfldl": ours,
        "current_season_nflreadpy": (
            nflreadpy_season
        ),
        "target_window": (
            get_settings().target_seasons
        ),
        "manifest_verified_on": (
            mf.verified_on
        ),
        "warnings": warnings,
        "datasets": {
            name: asdict(
                probe
            )
            for name, probe in probes.items()
        },
        "failures": sorted(
            name
            for name, probe in probes.items()
            if not probe.ok
        ),
    }

    COVERAGE_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COVERAGE_REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info(
        "coverage report -> %s",
        COVERAGE_REPORT_PATH,
    )

    return report


def write_back_to_manifest(
    report: dict[str, Any],
    path: Path | None = None,
) -> int:
    """Write only genuinely verified coverage into the YAML manifest.

    HEAD-derived ranges, sampled seasonal ranges, and failed schema checks are
    never allowed to overwrite committed ``verified_start`` or ``verified_end``
    fields.

    Returns the number of dataset entries eligible for write-back.
    """
    path = (
        path
        or MANIFEST_PATH
    )

    datasets = report[
        "datasets"
    ]

    eligible = {
        name: probe
        for name, probe in datasets.items()
        if (
            probe.get(
                "schema_verified"
            )
            is True
            and probe.get(
                "verified_start"
            )
            is not None
        )
    }

    if not eligible:
        log.info(
            "no content-backed verified coverage available; manifest unchanged"
        )

        return 0

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    current: str | None = None

    out: list[str] = []

    for line in lines:
        stripped = line.strip()

        if (
            line.startswith(
                "  "
            )
            and not line.startswith(
                "    "
            )
            and stripped.endswith(
                ":"
            )
        ):
            candidate = stripped[
                :-1
            ]

            if candidate in eligible:
                current = candidate
            else:
                current = None

        if (
            current
            and stripped.startswith(
                "verified_start:"
            )
        ):
            value = eligible[
                current
            ].get(
                "verified_start"
            )

            out.append(
                f"    verified_start: "
                f"{value if value is not None else 'null'}"
            )

            continue

        if (
            current
            and stripped.startswith(
                "verified_end:"
            )
        ):
            value = eligible[
                current
            ].get(
                "verified_end"
            )

            out.append(
                f"    verified_end: "
                f"{value if value is not None else 'null'}"
            )

            continue

        if stripped.startswith(
            "verified_on:"
        ):
            out.append(
                f'  verified_on: "{report["generated_at"][:10]}"'
            )

            continue

        out.append(
            line
        )

    path.write_text(
        "\n".join(
            out
        )
        + "\n",
        encoding="utf-8",
    )

    log.info(
        "manifest coverage fields updated in %s",
        path,
    )

    return len(
        eligible
    )


def _display_range(
    start: int | None,
    end: int | None,
    *,
    none_label: str = "NONE",
) -> str:
    if start is None:
        return none_label

    return (
        f"{start}-"
        f"{end if end is not None else 'current'}"
    )


def format_report(
    report: dict[str, Any],
) -> str:
    """Render verification evidence without calling HEAD results verified."""
    rows: list[str] = []

    for name, probe in sorted(
        report[
            "datasets"
        ].items()
    ):
        declared = _display_range(
            probe.get(
                "declared_start"
            ),
            probe.get(
                "declared_end"
            ),
            none_label="all-history",
        )

        asset_start = probe.get(
            "asset_start"
        )

        asset_end = probe.get(
            "asset_end"
        )

        asset_reachable = probe.get(
            "asset_reachable"
        )

        if asset_start is not None:
            reachable = _display_range(
                asset_start,
                asset_end,
            )

        elif asset_reachable is True:
            reachable = "single-file"

        elif asset_reachable is None:
            reachable = "n/a"

        else:
            reachable = "NONE"

        observed = _display_range(
            probe.get(
                "observed_start"
            ),
            probe.get(
                "observed_end"
            ),
        )

        verified = _display_range(
            probe.get(
                "verified_start"
            ),
            probe.get(
                "verified_end"
            ),
        )

        schema = (
            "yes"
            if probe.get(
                "schema_verified",
                False,
            )
            else "no"
        )

        gap = len(
            probe.get(
                "target_window_gap",
                [],
            )
        )

        flag = (
            " "
            if probe.get(
                "ok",
                True,
            )
            else "!"
        )

        rows.append(
            f"{flag} {name:18s} "
            f"declared={declared:14s} "
            f"reachable={reachable:12s} "
            f"observed={observed:12s} "
            f"schema={schema:3s} "
            f"verified={verified:12s} "
            f"target_gap={gap}"
        )

    generated_at = report.get(
        "generated_at",
        "unknown",
    )

    current_season = report.get(
        "current_season_nfldl",
        "unknown",
    )

    header = (
        f"provider verification {generated_at[:19]}Z "
        f"(current season: {current_season})"
    )

    body = "\n".join(
        rows
    )

    tail = ""

    warnings = report.get(
        "warnings",
        [],
    )

    if warnings:
        tail += (
            "\n\nwarnings:\n"
            + "\n".join(
                f"  - {warning}"
                for warning in warnings
            )
        )

    failures = report.get(
        "failures",
        [],
    )

    if failures:
        tail += (
            "\n\nFAILED: "
            + ", ".join(
                failures
            )
        )

    return (
        f"{header}\n"
        f"{'-' * len(header)}\n"
        f"{body}"
        f"{tail}"
    )