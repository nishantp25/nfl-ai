"""Refresh current-season data and rebuild forecasting candidates."""
import os
os.environ["NFLREADPY_CACHE"] = "off"
os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NFLDL_USER_AGENT"] = "nfl-ai/1.0 (contact: nishantsownworld@gmail.com)"
os.environ["NFLREADPY_USER_AGENT"] = os.environ["NFLDL_USER_AGENT"]

import fcntl
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
from nfldl.config import nfl_season_of
from nfldl.manifest import get_manifest
from nfldl.quality import CHECKS
from nfldl.storage import read_dataset, _atomic_write_parquet

ROOT = Path(__file__).resolve().parents[1]

def run(arguments, timeout=900):
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True, timeout=timeout)

def validate(frame, spec):
    errors = [r.detail for check in CHECKS
              if not (r := check(frame, spec)).passed and r.severity == "error"]
    if errors:
        raise RuntimeError({spec.name: errors})

def main():
    os.chdir(ROOT)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with (reports / "weather_refresh.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another data refresh is running; skipped.", flush=True)
            return
        season = nfl_season_of()
        names = ["schedules", "play_by_play", "player_stats", "team_stats",
                 "snap_counts", "rosters", "depth_charts", "injuries",
                 "weekly_roster_snapshots"]
        run(["-m", "nfldl.cli", "ingest", *names, "-s", str(season), "--force"])
        mf = get_manifest()
        for name in [*names, "weekly_rosters"]:
            frame = read_dataset(mf[name], seasons=[season])
            if name == "player_stats":
                identified = frame.filter(pl.col("player_id").is_not_null())
                unresolved = frame.filter(pl.col("player_id").is_null())
                validate(identified, mf[name])
                for folder, part in (("curated", identified), ("quarantine", unresolved)):
                    target = ROOT / "data" / folder / name / f"season={season}/part.parquet"
                    _atomic_write_parquet(part, target)
                print(f"Player stats: {identified.height} identified, {unresolved.height} quarantined.",
                      flush=True)
            else:
                validate(frame, mf[name])
        run([str(ROOT / "scripts/build_team_performance.py")], timeout=300)
        digest = hashlib.sha256(b"live-training-v1")
        schedules = read_dataset(mf["schedules"], seasons=list(range(2011, season + 1)))
        fields = ["game_id", "season", "kickoff_utc", "home_team", "away_team",
                  "home_score", "away_score", "location"]
        digest.update(schedules.select(fields).sort("game_id").write_json().encode())
        for path in sorted((ROOT / "data/curated/team_performance").glob("season=*/part.parquet")):
            frame = pl.read_parquet(path).sort(["game_id", "team"])
            digest.update(str(path.relative_to(ROOT)).encode())
            float_columns = [
                name for name, dtype in frame.schema.items()
                if dtype in (pl.Float32, pl.Float64)
            ]
            canonical = frame.with_columns(
                [pl.col(name).round(8) for name in float_columns]
            ).select(sorted(frame.columns))
            digest.update(canonical.write_json().encode())
        now = datetime.now(timezone.utc)
        eligible = schedules.filter(
            pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
            & ((pl.col("kickoff_utc") + pl.duration(hours=12)) <= pl.lit(now))
        ).select("game_id").sort("game_id")
        upcoming = schedules.filter(
            pl.col("kickoff_utc") > pl.lit(now)
        ).select("game_id").sort("game_id")
        digest.update(eligible.write_json().encode())
        digest.update(upcoming.write_json().encode())
        fingerprint = digest.hexdigest()
        target = reports / "core_refresh.json"
        previous = json.loads(target.read_text()) if target.exists() else {}
        changed = previous.get("input_sha256") != fingerprint
        if changed:
            run([str(ROOT / "scripts/train_live.py")], timeout=600)
        payload = {
            "last_success_utc": datetime.now(timezone.utc).isoformat(),
            "season": season, "input_sha256": fingerprint,
            "candidate_rebuilt": changed,
            "status": "validated; candidates only; no automatic production promotion",
            "training_window": "Live fit: all eligible completed games since 2012; evaluation fit: 2012-2023",
        }
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(target)
        print(json.dumps(payload, indent=2), flush=True)

if __name__ == "__main__":
    main()
