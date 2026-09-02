"""Verify what actually exists before ingesting anything.

The brief is explicit: *do not assume every dataset covers 2011-2026*. This
module is how that instruction is enforced mechanically rather than by good
intentions.

For every dataset it establishes, against the live provider:

* which seasons actually have an asset (HEAD probes -- no payload downloaded);
* the real schema and row count for one sampled season;
* whether ``nflreadpy``'s own season heuristic agrees with reality.

Output goes to ``reports/verified_coverage.json``, which the manifest overlays
on load, so the rest of the codebase reads verified numbers automatically.
``--write-manifest`` writes the discovered ranges back into
``configs/data_sources.yaml`` so the committed manifest never drifts from truth.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .config import COVERAGE_REPORT_PATH, MANIFEST_PATH, get_settings, nfl_season_of
from .http import head_ok
from .manifest import DatasetSpec, get_manifest
from .registry import REGISTRY, get_adapter

log = logging.getLogger(__name__)

# How far past the declared start / current season to look. Probing a little
# wider than expected is how you discover a backfill you did not know about.
PROBE_MARGIN_BACK = 3
PROBE_MARGIN_FORWARD = 1
# Nothing in nflverse predates 1999 in a form we would model on.
PROBE_FLOOR = 1999


@dataclass
class DatasetProbe:
    dataset: str
    provider: str
    declared_start: int | None
    declared_end: int | None
    verified_start: int | None = None
    verified_end: int | None = None
    seasons_present: list[int] = field(default_factory=list)
    seasons_missing_in_declared_range: list[int] = field(default_factory=list)
    target_window_gap: list[int] = field(default_factory=list)
    probe_method: str = "head"
    sampled_season: int | None = None
    sampled_rows: int | None = None
    sampled_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ok: bool = True


def _probe_seasons(spec: DatasetSpec, current_season: int) -> tuple[list[int], int]:
    """Returns (seasons found, earliest season actually probed)."""
    adapter = get_adapter(spec.name)
    if adapter.asset_url is None:
        return [], current_season

    # Probe a little wider than declared, but never below the earliest season
    # any nflverse asset covers -- and never below what the project targets, so
    # a 1920-declared roster file is not probed back to 1920 pointlessly.
    floor = max(PROBE_FLOOR, min(get_settings().target_season_start, spec.declared.start or PROBE_FLOOR))
    start = max((spec.declared.start or current_season) - PROBE_MARGIN_BACK, floor)
    end = current_season + PROBE_MARGIN_FORWARD
    candidates = list(range(start, end + 1))

    variants = spec.variants if get_adapter(spec.name).variant_assets else (None,)

    def present(season: int) -> int | None:
        for variant in variants or (None,):
            keys: dict[str, Any] = {"season": season}
            if variant is not None:
                keys["stat_type"] = variant
            url = adapter.asset_url(**keys)
            if url and head_ok(url):
                return season
        return None

    with ThreadPoolExecutor(max_workers=get_settings().max_workers) as pool:
        found = list(pool.map(present, candidates))
    return sorted(s for s in found if s is not None), candidates[0]


def probe_dataset(name: str, *, sample: bool = False) -> DatasetProbe:
    mf = get_manifest()
    spec = mf[name]
    adapter = get_adapter(name)
    cfg = get_settings()
    current = nfl_season_of()

    probe = DatasetProbe(
        dataset=name,
        provider=spec.provider,
        declared_start=spec.declared.start,
        declared_end=spec.declared.end,
    )

    # Datasets we build locally (stadiums, coaching, weather) have no remote
    # asset to probe; their coverage is a function of their inputs.
    if adapter.asset_url is None:
        probe.probe_method = "derived"
        probe.notes.append(
            f"built locally from {list(adapter.needs_local) or 'curated config'}; "
            "coverage follows its inputs"
        )
        probe.verified_start = spec.declared.start
        probe.verified_end = spec.declared.end
        return probe

    if not adapter.seasonal_assets:
        keys: dict[str, Any] = {}
        if adapter.variant_assets and spec.variants:
            keys["stat_type"] = spec.variants[0]
        url = adapter.asset_url(**keys)
        probe.probe_method = "head-single-file"
        probe.ok = bool(url and head_ok(url))
        if not probe.ok:
            probe.notes.append(f"asset not reachable: {url}")
        elif sample:
            probe.verified_start = spec.declared.start
            probe.verified_end = spec.declared.end
            _sample_single_file(spec, probe)
        else:
            probe.verified_start = spec.declared.start
            probe.verified_end = spec.declared.end
            probe.probe_method = "head-single-file"
            probe.notes.append(
                "single all-history asset: reachable, but the season range inside it is the "
                "provider's claim, not something a HEAD request can confirm. Use "
                "`nfldl verify --sample` to check the real min/max season."
            )
        return probe

    present, start_probed = _probe_seasons(spec, current)
    probe.seasons_present = present
    if not present:
        probe.ok = False
        probe.notes.append("no seasonal assets found -- release tag or filename pattern may have changed")
        return probe

    probe.verified_start, probe.verified_end = present[0], present[-1]

    contiguous = set(range(present[0], present[-1] + 1))
    holes = sorted(contiguous - set(present))
    if holes:
        probe.seasons_missing_in_declared_range = holes
        probe.notes.append(f"non-contiguous coverage; missing {holes}")

    probed_floor = min(candidates_probed) if (candidates_probed := probe.seasons_present) else None
    declared_start = spec.declared.start
    if declared_start is not None and probe.verified_start > declared_start:
        if probe.verified_start > start_probed:
            # We looked earlier than the first hit and found nothing: real gap.
            probe.ok = False
            probe.notes.append(
                f"manifest claims data from {declared_start} but nothing exists before "
                f"{probe.verified_start}"
            )
        else:
            probe.notes.append(
                f"declared start is {declared_start}; probing began at {start_probed} "
                f"(project floor), so earlier seasons were not checked"
            )

    wanted = set(cfg.target_seasons)
    gap = sorted(wanted - set(present))
    if gap:
        probe.target_window_gap = gap
        probe.notes.append(
            f"{len(gap)} of the {len(wanted)} target seasons are unavailable: {gap}. "
            "Features from this dataset must be null there, not imputed."
        )

    if sample:
        season = present[-1]
        try:
            kwargs: dict[str, Any] = {"season": season}
            if adapter.variant_assets and spec.variants:
                kwargs["stat_type"] = spec.variants[0]
            df = adapter.fetch(**kwargs)
            probe.sampled_season = season
            probe.sampled_rows = df.height
            probe.sampled_columns = df.columns
            _check_primary_key(df, spec, probe)
        except Exception as exc:
            probe.ok = False
            probe.notes.append(f"sample fetch for {season} failed: {exc}")

    return probe


def _sample_single_file(spec: DatasetSpec, probe: DatasetProbe) -> None:
    """Download a single-file dataset once to learn its true season range."""
    adapter = get_adapter(spec.name)
    kwargs: dict[str, Any] = {}
    if adapter.variant_assets and spec.variants:
        kwargs["stat_type"] = spec.variants[0]
    try:
        df = adapter.fetch(**kwargs)
    except Exception as exc:
        probe.ok = False
        probe.notes.append(f"sample fetch failed: {exc}")
        return
    probe.sampled_rows = df.height
    probe.sampled_columns = df.columns
    season_col = next((c for c in ("season", "year_signed", "draft_year") if c in df.columns), None)
    if season_col:
        seasons = sorted({int(s) for s in df.get_column(season_col).drop_nulls().to_list()})
        probe.seasons_present = seasons
        probe.verified_start, probe.verified_end = seasons[0], seasons[-1]
        probe.probe_method = "sampled-single-file"
    _check_primary_key(df, spec, probe)


def _check_primary_key(df: pl.DataFrame, spec: DatasetSpec, probe: DatasetProbe) -> None:
    pk = [c for c in spec.primary_key if c in df.columns]
    absent = [c for c in spec.primary_key if c not in df.columns]
    if absent:
        probe.ok = False
        probe.notes.append(
            f"declared primary_key columns absent from the real schema: {absent}. "
            "Fix the manifest -- a wrong key silently permits duplicates."
        )
    if pk and len(pk) == len(spec.primary_key):
        dupes = df.height - df.select(pk).unique().height
        if dupes:
            probe.ok = False
            probe.notes.append(f"primary key {pk} is not unique: {dupes} duplicate rows")


def verify_all(names: list[str] | None = None, *, sample: bool = False) -> dict[str, Any]:
    mf = get_manifest()
    names = names or sorted(REGISTRY)
    probes = {}
    for name in names:
        log.info("probing %s", name)
        probes[name] = probe_dataset(name, sample=sample)

    nflreadpy_season = None
    try:
        import nflreadpy as nfl

        nflreadpy_season = nfl.get_current_season()
    except Exception:  # pragma: no cover
        pass

    ours = nfl_season_of()
    warnings: list[str] = []
    if nflreadpy_season is not None and nflreadpy_season != ours:
        warnings.append(
            f"season heuristics disagree: nfldl says {ours}, nflreadpy says {nflreadpy_season}. "
            "Expect nflreadpy to reject in-range seasons near the rollover; the direct-asset "
            "fallback in sources.nflverse covers this."
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_season_nfldl": ours,
        "current_season_nflreadpy": nflreadpy_season,
        "target_window": get_settings().target_seasons,
        "manifest_verified_on": mf.verified_on,
        "warnings": warnings,
        "datasets": {k: asdict(v) for k, v in probes.items()},
        "failures": sorted(k for k, v in probes.items() if not v.ok),
    }

    COVERAGE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_REPORT_PATH.write_text(json.dumps(report, indent=2))
    log.info("coverage report -> %s", COVERAGE_REPORT_PATH)
    return report


def write_back_to_manifest(report: dict[str, Any], path: Path | None = None) -> None:
    """Rewrite verified_start/verified_end in the YAML, preserving comments.

    Deliberately a line-level edit rather than a YAML round-trip: the manifest's
    comments carry the caveats, and a naive dump would delete all of them.
    """
    path = path or MANIFEST_PATH
    lines = path.read_text().splitlines()
    datasets = report["datasets"]

    current: str | None = None
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            candidate = stripped[:-1]
            if candidate in datasets:
                current = candidate
            elif candidate not in ("defaults", "manifest"):
                current = None
        if current and stripped.startswith("verified_start:"):
            v = datasets[current].get("verified_start")
            out.append(f"    verified_start: {v if v is not None else 'null'}")
            continue
        if current and stripped.startswith("verified_end:"):
            v = datasets[current].get("verified_end")
            out.append(f"    verified_end: {v if v is not None else 'null'}")
            continue
        if stripped.startswith("verified_on:"):
            out.append(f'  verified_on: "{report["generated_at"][:10]}"')
            continue
        out.append(line)

    path.write_text("\n".join(out) + "\n")
    log.info("manifest coverage fields updated in %s", path)


def format_report(report: dict[str, Any]) -> str:
    rows = []
    for name, p in sorted(report["datasets"].items()):
        declared = f"{p['declared_start']}-{p['declared_end'] or 'current'}"
        if p["verified_start"] is None:
            verified = "NONE" if not p["ok"] else "all-history"
        else:
            verified = f"{p['verified_start']}-{p['verified_end'] or 'current'}"
        gap = len(p["target_window_gap"])
        flag = " " if p["ok"] else "!"
        rows.append(f"{flag} {name:18s} declared={declared:14s} verified={verified:12s} target_gap={gap}")
    header = (
        f"coverage verified {report['generated_at'][:19]}Z  "
        f"(current season: {report['current_season_nfldl']})"
    )
    body = "\n".join(rows)
    tail = ""
    if report["warnings"]:
        tail += "\n\nwarnings:\n" + "\n".join(f"  - {w}" for w in report["warnings"])
    if report["failures"]:
        tail += "\n\nFAILED: " + ", ".join(report["failures"])
    return f"{header}\n{'-' * len(header)}\n{body}{tail}"
