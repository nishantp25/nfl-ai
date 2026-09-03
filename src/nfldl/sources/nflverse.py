"""nflverse adapters -- the primary source for everything nflverse provides.

Two things this module does that a thin ``nflreadpy`` wrapper would not:

1. **Asset URLs for probing.** Every seasonal dataset knows the exact release
   asset it maps to, so ``nfldl verify`` can establish real coverage with HEAD
   requests instead of downloading gigabytes or trusting a hard-coded range.

2. **A guarded fallback.** ``nflreadpy`` validates the requested season against
   a *date heuristic* for "current season" and raises ``ValueError`` outside it.
   Around the September rollover, and for league-year assets that publish ahead
   of the playing season, that heuristic rejects seasons whose files genuinely
   exist. When -- and only when -- the asset HEAD-probes as present, we read the
   identical release asset directly. Same provider, same file, same bytes; we
   are routing around a client-side date check, not around the source.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from typing import Any

import nflreadpy as nfl
import polars as pl

from ..http import head_ok, request

log = logging.getLogger(__name__)

RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# dataset -> (release_tag, file stem template). "{season}" marks a seasonal
# asset; a stem without it is a single all-history file.
ASSETS: dict[str, tuple[str, str]] = {
    "play_by_play": ("pbp", "play_by_play_{season}"),
    "schedules": ("schedules", "games"),
    "teams": ("teams", "teams_colors_logos"),
    "players": ("players", "players"),
    "rosters": ("rosters", "roster_{season}"),
    "weekly_rosters": ("weekly_rosters", "roster_weekly_{season}"),
    "depth_charts": ("depth_charts", "depth_charts_{season}"),
    "injuries": ("injuries", "injuries_{season}"),
    "snap_counts": ("snap_counts", "snap_counts_{season}"),
    "player_stats": ("stats_player", "stats_player_week_{season}"),
    "team_stats": ("stats_team", "stats_team_week_{season}"),
    "participation": ("pbp_participation", "pbp_participation_{season}"),
    "ftn_charting": ("ftn_charting", "ftn_charting_{season}"),
    "nextgen_stats": ("nextgen_stats", "ngs_{stat_type}"),
    "draft_picks": ("draft_picks", "draft_picks"),
    "trades": ("trades", "trades"),
    "contracts": ("contracts", "historical_contracts"),
    "officials": ("officials", "officials"),
    "combine": ("combine", "combine"),
    "pfr_advstats": ("pfr_advstats", "advstats_week_{stat_type}_{season}"),
}


def is_seasonal(dataset: str) -> bool:
    """True if this dataset publishes one asset per season."""
    entry = ASSETS.get(dataset)
    return bool(entry) and "{season}" in entry[1]


def needs_variant(dataset: str) -> bool:
    """True if the asset name is parameterised by stat_type."""
    entry = ASSETS.get(dataset)
    return bool(entry) and "{stat_type}" in entry[1]


def asset_url(dataset: str, *, season: int | None = None, stat_type: str | None = None) -> str | None:
    """Fully-qualified release-asset URL, or ``None`` if we cannot address it."""
    entry = ASSETS.get(dataset)
    if entry is None:
        return None
    tag, stem = entry
    if "{season}" in stem and season is None:
        return None
    if "{stat_type}" in stem and stat_type is None:
        return None
    stem = stem.format(season=season, stat_type=stat_type)
    return f"{RELEASE_BASE}/{tag}/{stem}.parquet"


def _read_asset_directly(url: str) -> pl.DataFrame:
    resp = request("GET", url)
    resp.raise_for_status()
    return pl.read_parquet(io.BytesIO(resp.content))


def _with_fallback(
    dataset: str,
    loader: Callable[[], pl.DataFrame],
    *,
    season: int | None = None,
    stat_type: str | None = None,
) -> pl.DataFrame:
    """Run the nflreadpy loader; on a *validation* error, use the asset directly."""
    try:
        return loader()
    except ValueError as exc:
        url = asset_url(dataset, season=season, stat_type=stat_type)
        if url is None or not head_ok(url):
            raise
        log.warning(
            "nflreadpy rejected %s season=%s (%s) but the release asset exists; "
            "reading %s directly",
            dataset, season, exc, url,
        )
        df = _read_asset_directly(url)
        if season is not None and "season" in df.columns:
            df = df.filter(pl.col("season") == season)
        return df


def _season_filter(df: pl.DataFrame, season: int | None) -> pl.DataFrame:
    if season is None or "season" not in df.columns:
        return df
    return df.filter(pl.col("season") == season)


# ---------------------------------------------------------------------------
# Fetchers. Signature is uniform: (season: int | None, **kwargs) -> DataFrame
# ---------------------------------------------------------------------------

def fetch_play_by_play(season: int, **_: Any) -> pl.DataFrame:
    return _with_fallback("play_by_play", lambda: nfl.load_pbp(season), season=season)


def fetch_schedules(season: int, **_: Any) -> pl.DataFrame:
    """Schedules ship as one all-history file; we partition it ourselves.

    We also derive ``kickoff_utc`` here rather than downstream, because it is the
    anchor for every point-in-time guard and for the weather join, and deriving
    it in one place stops three modules from disagreeing about timezone.
    """
    df = _with_fallback("schedules", lambda: nfl.load_schedules(True))
    df = _season_filter(df, season)
    return _add_kickoff_utc(df)


def _add_kickoff_utc(df: pl.DataFrame) -> pl.DataFrame:
    """Combine gameday + gametime into a UTC timestamp.

    nflverse gametime is US/Eastern local. Games in London, Munich, Mexico City
    and São Paulo are still stamped Eastern in this field, so a single Eastern
    interpretation is correct for all of them.
    """
    if "gameday" not in df.columns:
        return df.with_columns(pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("kickoff_utc"))

    gametime = (
        pl.col("gametime").fill_null("13:00") if "gametime" in df.columns else pl.lit("13:00")
    )
    naive = (
        pl.concat_str([pl.col("gameday").cast(pl.String), pl.lit(" "), gametime])
        .str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M", strict=False)
    )
    return df.with_columns(
        naive.dt.replace_time_zone("America/New_York", ambiguous="earliest")
        .dt.convert_time_zone("UTC")
        .alias("kickoff_utc")
    )


def fetch_teams(**_: Any) -> pl.DataFrame:
    return nfl.load_teams()


def fetch_players(**_: Any) -> pl.DataFrame:
    return nfl.load_players()


def fetch_rosters(season: int, **_: Any) -> pl.DataFrame:
    return _with_fallback("rosters", lambda: nfl.load_rosters(season), season=season)


def fetch_weekly_rosters(season: int, **_: Any) -> pl.DataFrame:
    return _with_fallback("weekly_rosters", lambda: nfl.load_rosters_weekly(season), season=season)


def fetch_depth_charts(season: int, **_: Any) -> pl.DataFrame:
    """Fetch depth charts and remove exact duplicate upstream rows."""
    df = _with_fallback(
        "depth_charts",
        lambda: nfl.load_depth_charts(season),
        season=season,
    )
    return df.unique(maintain_order=True)


def fetch_injuries(season: int, **_: Any) -> pl.DataFrame:
    return _with_fallback("injuries", lambda: nfl.load_injuries(season), season=season)


def fetch_snap_counts(season: int, **_: Any) -> pl.DataFrame:
    return _with_fallback("snap_counts", lambda: nfl.load_snap_counts(season), season=season)


def fetch_player_stats(season: int, **_: Any) -> pl.DataFrame:
    return _with_fallback(
        "player_stats",
        lambda: nfl.load_player_stats(season, summary_level="week"),
        season=season,
    )


def fetch_team_stats(season: int, **_: Any) -> pl.DataFrame:
    return _with_fallback(
        "team_stats",
        lambda: nfl.load_team_stats(season, summary_level="week"),
        season=season,
    )


def fetch_participation(season: int, **_: Any) -> pl.DataFrame:
    return _with_fallback("participation", lambda: nfl.load_participation(season), season=season)


def fetch_ftn_charting(season: int, **_: Any) -> pl.DataFrame:
    return _with_fallback("ftn_charting", lambda: nfl.load_ftn_charting(season), season=season)


def fetch_nextgen_stats(season: int, stat_type: str = "passing", **_: Any) -> pl.DataFrame:
    df = _with_fallback(
        "nextgen_stats",
        lambda: nfl.load_nextgen_stats(season, stat_type=stat_type),
        season=season,
        stat_type=stat_type,
    )
    df = _season_filter(df, season)
    return df.with_columns(pl.lit(stat_type).alias("stat_type"))


def fetch_pfr_advstats(season: int, stat_type: str = "pass", **_: Any) -> pl.DataFrame:
    df = _with_fallback(
        "pfr_advstats",
        lambda: nfl.load_pfr_advstats(season, stat_type=stat_type, summary_level="week"),
        season=season,
        stat_type=stat_type,
    )
    return df.with_columns(pl.lit(stat_type).alias("stat_type"))


def fetch_draft_picks(**_: Any) -> pl.DataFrame:
    return nfl.load_draft_picks(True)


def fetch_trades(**_: Any) -> pl.DataFrame:
    return nfl.load_trades()


def fetch_contracts(**_: Any) -> pl.DataFrame:
    return nfl.load_contracts()


def fetch_officials(**_: Any) -> pl.DataFrame:
    return nfl.load_officials(True)


def fetch_combine(**_: Any) -> pl.DataFrame:
    return nfl.load_combine(True)
