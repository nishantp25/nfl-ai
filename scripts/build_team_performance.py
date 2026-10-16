"""Build postgame team statistics; downstream features must lag these."""
import os
os.environ["POLARS_MAX_THREADS"] = "2"
from pathlib import Path
import polars as pl
from nfldl.config import RAW_DIR, nfl_season_of
from nfldl.manifest import get_manifest

ROOT = Path(__file__).resolve().parents[1]

def main():
    spec = get_manifest()["play_by_play"]
    for season in range(2011, nfl_season_of() + 1):
        source = spec.storage_path(RAW_DIR, season=season)
        plays = pl.scan_parquet(source).select(
            "game_id", "posteam", "defteam", "play_type", "epa"
        ).filter(
            pl.col("play_type").is_in(["pass", "run"])
            & pl.col("epa").is_not_null()
            & pl.col("posteam").is_not_null()
            & pl.col("defteam").is_not_null()
        )
        offense = plays.group_by("game_id", "posteam").agg(
            pl.col("epa").mean().alias("off_epa"),
            (pl.col("epa") > 0).mean().alias("off_success"),
            (pl.col("play_type") == "pass").mean().alias("pass_rate"),
            pl.len().alias("off_plays"),
        ).rename({"posteam": "team"})
        defense = plays.group_by("game_id", "defteam").agg(
            pl.col("epa").mean().alias("def_epa_allowed"),
            (pl.col("epa") > 0).mean().alias("def_success_allowed"),
            pl.len().alias("def_plays"),
        ).rename({"defteam": "team"})
        frame = offense.join(defense, on=["game_id", "team"], how="full", coalesce=True).collect()
        assert frame.select("game_id", "team").unique().height == frame.height
        frame = frame.with_columns(pl.lit(season).alias("season"))
        target = ROOT / "data/curated/team_performance" / f"season={season}/part.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary)
        temporary.replace(target)
        print(f"{season}: {frame.height:,} team-game rows", flush=True)

if __name__ == "__main__":
    main()
