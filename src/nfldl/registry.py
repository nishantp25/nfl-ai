"""The registry: the only place a dataset name becomes executable code.

Two invariants are enforced at import time, and violating either is a hard
failure rather than a warning:

* **No undeclared data.** Every fetcher must correspond to a dataset in
  ``configs/data_sources.yaml``. You cannot quietly add a convenient dataset.
* **No downloaded derived features.** Registering a fetcher for anything on the
  manifest's ``forbidden_downloads`` list raises. Elo, opponent-adjusted EPA,
  strength of schedule and the rest are computed from canonical raw data, so
  they can be rebuilt for any historical date without leaking the future.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

from .manifest import DatasetSpec, ManifestError, get_manifest
from .sources import coaching, nflverse, stadiums, weather, weather_forecast

Fetcher = Callable[..., pl.DataFrame]


@dataclass(frozen=True)
class Adapter:
    """Binds a manifest dataset to the code that fetches it."""

    name: str
    fetch: Fetcher
    asset_url: Callable[..., str | None] | None = None
    needs_local: tuple[str, ...] = ()  # datasets that must be ingested first
    seasonal_assets: bool = False      # one remote asset per season?
    variant_assets: bool = False       # asset name parameterised by stat_type?

    @property
    def spec(self) -> DatasetSpec:
        return get_manifest()[self.name]


def _nflverse_url(name: str) -> Callable[..., str | None]:
    def _url(**keys: Any) -> str | None:
        return nflverse.asset_url(name, **keys)

    return _url


def _nflverse(name: str, fetch: Fetcher, **kw: Any) -> Adapter:
    """Build an adapter that knows the shape of its remote asset."""
    return Adapter(
        name,
        fetch,
        _nflverse_url(name),
        seasonal_assets=nflverse.is_seasonal(name),
        variant_assets=nflverse.needs_variant(name),
        **kw,
    )


_ADAPTERS: list[Adapter] = [
    # ---- nflverse canonical -------------------------------------------
    _nflverse("play_by_play", nflverse.fetch_play_by_play),
    _nflverse("schedules", nflverse.fetch_schedules),
    _nflverse("teams", nflverse.fetch_teams),
    _nflverse("players", nflverse.fetch_players),
    _nflverse("rosters", nflverse.fetch_rosters),
    _nflverse("weekly_rosters", nflverse.fetch_weekly_rosters),
    _nflverse("depth_charts", nflverse.fetch_depth_charts),
    _nflverse("injuries", nflverse.fetch_injuries),
    _nflverse("snap_counts", nflverse.fetch_snap_counts),
    _nflverse("player_stats", nflverse.fetch_player_stats),
    _nflverse("team_stats", nflverse.fetch_team_stats),
    _nflverse("participation", nflverse.fetch_participation),
    _nflverse("ftn_charting", nflverse.fetch_ftn_charting),
    _nflverse("nextgen_stats", nflverse.fetch_nextgen_stats),
    _nflverse("pfr_advstats", nflverse.fetch_pfr_advstats),
    _nflverse("draft_picks", nflverse.fetch_draft_picks),
    _nflverse("trades", nflverse.fetch_trades),
    _nflverse("contracts", nflverse.fetch_contracts),
    _nflverse("officials", nflverse.fetch_officials),
    _nflverse("combine", nflverse.fetch_combine),
    # ---- supplementary -------------------------------------------------
    Adapter("stadiums", stadiums.build, needs_local=("schedules",)),
    Adapter("coaching_staff", coaching.build),
    Adapter("weather", weather.build, needs_local=("schedules", "stadiums")),
    Adapter(
        "weather_forecast",
        weather_forecast.build,
        needs_local=("schedules", "stadiums"),
    ),
]

REGISTRY: dict[str, Adapter] = {a.name: a for a in _ADAPTERS}


def _validate_registry() -> None:
    mf = get_manifest()
    declared = set(mf.names())
    registered = set(REGISTRY)

    for name in registered:
        mf.assert_not_forbidden(name)

    undeclared = registered - declared
    if undeclared:
        raise ManifestError(
            f"fetchers exist for undeclared datasets {sorted(undeclared)}. Every raw dataset "
            f"must be documented in {mf.path.name} before any code may download it."
        )

    unimplemented = declared - registered
    if unimplemented:
        raise ManifestError(
            f"manifest declares {sorted(unimplemented)} with no fetcher. Either implement one "
            "or remove the entry -- an undeliverable manifest entry is a lie."
        )

    for name, adapter in REGISTRY.items():
        for dep in adapter.needs_local:
            if dep not in declared:
                raise ManifestError(f"'{name}' depends on undeclared dataset '{dep}'")


_validate_registry()


def get_adapter(name: str) -> Adapter:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ManifestError(
            f"unknown dataset '{name}'. Known: {sorted(REGISTRY)}"
        ) from None


def resolve_order(names: list[str]) -> list[str]:
    """Topologically order datasets so dependencies are ingested first."""
    ordered: list[str] = []
    seen: set[str] = set()
    visiting: set[str] = set()

    def visit(n: str) -> None:
        if n in seen:
            return
        if n in visiting:
            raise ManifestError(f"circular dependency involving '{n}'")
        visiting.add(n)
        for dep in get_adapter(n).needs_local:
            visit(dep)
        visiting.discard(n)
        seen.add(n)
        ordered.append(n)

    for name in names:
        visit(name)
    return ordered
