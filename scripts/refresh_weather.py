"""Refresh and validate live game-weather snapshots."""
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["NFLREADPY_CACHE"] = "off"
os.environ["NFLDL_USER_AGENT"] = "nfl-ai/1.0 (contact: nishantsownworld@gmail.com)"
os.environ["NFLREADPY_USER_AGENT"] = os.environ["NFLDL_USER_AGENT"]
os.environ["POLARS_MAX_THREADS"] = "2"

from nfldl.config import nfl_season_of
from nfldl.manifest import get_manifest
from nfldl.quality import CHECKS
from nfldl.storage import read_dataset

def main():
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with (reports / "weather_refresh.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Weather refresh already running; skipped.", flush=True)
            return
        season = nfl_season_of()
        subprocess.run(
            [sys.executable, "-m", "nfldl.cli", "ingest", "weather_forecast",
             "-s", str(season), "--force"],
            check=True, timeout=900,
        )
        spec = get_manifest()["weather_forecast"]
        frame = read_dataset(spec, seasons=[season])
        errors = [r.detail for check in CHECKS
                  if not (r := check(frame, spec)).passed and r.severity == "error"]
        if errors:
            raise RuntimeError(errors)
        latest = frame.sort("retrieved_at_utc").unique(
            subset=["game_id"], keep="last", maintain_order=True)
        fields = ["game_id", "kickoff_utc", "temperature_c", "wind_speed_kmh",
                  "wind_gust_kmh", "precipitation_probability_pct",
                  "precipitation_mm", "weather_method", "roof_unknown"]
        games = {row["game_id"]: row for row in latest.select(fields).to_dicts()}
        target = reports / "weather_latest.json"
        previous = json.loads(target.read_text()).get("games", {}) if target.exists() else {}
        comparable = json.loads(json.dumps(games, default=str))
        changed = [key for key, value in comparable.items() if previous.get(key) != value]
        payload = {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "season": season, "changed_game_ids": changed,
            "games": comparable, "prediction_status": "model_not_implemented",
        }
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(target)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_weather_accuracy.py")],
            check=True, timeout=120,
        )
        print(f"Validated {len(games)} game forecasts; {len(changed)} new or changed.",
              flush=True)

if __name__ == "__main__":
    main()
