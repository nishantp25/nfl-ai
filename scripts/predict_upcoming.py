"""Predict only games starting within 24 hours, using a saved live candidate."""
import os
os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import fcntl
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib
import polars as pl
from train_live import build_rows, matrix, FEATURES
from performance_features import enrich
from nfldl.config import nfl_season_of
from nfldl.manifest import get_manifest
from nfldl.storage import read_dataset

ROOT = Path(__file__).resolve().parents[1]

def main():
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with (reports / "weather_refresh.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Data refresh in progress; prediction run skipped.", flush=True)
            return
        now = datetime.now(timezone.utc)
        health = json.loads((reports / "core_refresh.json").read_text())
        last_success = datetime.fromisoformat(health["last_success_utc"])
        age = (now - last_success).total_seconds()
        if not 0 <= age <= 3 * 3600:
            raise RuntimeError("Core data is stale; refresh it before predicting.")
        candidates = [
            path for path in (ROOT / "data/models").glob("live_*")
            if (path / "model.joblib").exists() and (path / "report.json").exists()
        ]
        if not candidates:
            raise RuntimeError("No complete live candidate found.")
        candidate = max(candidates, key=lambda path: path.name)
        saved = joblib.load(candidate / "model.joblib")
        assert saved["features"] == FEATURES, "Model/feature mismatch"
        schedules = read_dataset(
            get_manifest()["schedules"], seasons=list(range(2011, nfl_season_of() + 1))
        ).filter(pl.col("kickoff_utc").is_not_null()).to_dicts()
        performance = []
        for season in range(2011, nfl_season_of() + 1):
            path = ROOT / "data/curated/team_performance" / f"season={season}/part.parquet"
            performance.extend(pl.read_parquet(path).to_dicts())
        rows = enrich(build_rows(schedules, now), performance, schedules, now)
        upcoming = [
            row for row in rows
            if now < row["kickoff_utc"] <= now + timedelta(hours=24)
        ]
        games = {game["game_id"]: game for game in schedules}
        predictions = []
        if upcoming:
            probabilities = saved["model"].predict_proba(matrix(upcoming))[:, 1]
            for row, probability in zip(upcoming, probabilities):
                game = games[row["game_id"]]
                predictions.append({
                    "game_id": row["game_id"],
                    "home_team": game["home_team"], "away_team": game["away_team"],
                    "kickoff_utc": row["kickoff_utc"].isoformat(),
                    "home_win_probability_conditional_on_no_tie": float(probability),
                    "feature_values": {name: row[name] for name in FEATURES},
                })
        payload = {
            "generated_at_utc": now.isoformat(),
            "core_data_last_success_utc": last_success.isoformat(),
            "model_directory": str(candidate), "horizon_hours": 24,
            "weather_and_injuries_in_model": False,
            "predictions": predictions,
        }
        # Convert missing numeric feature values to JSON null.
        for prediction in predictions:
            prediction["feature_values"] = {
                key: None if value != value else value
                for key, value in prediction["feature_values"].items()
            }
        content = json.dumps(payload, indent=2, allow_nan=False)
        archive = reports / "predictions"
        archive.mkdir(exist_ok=True)
        (archive / now.strftime("%Y%m%dT%H%M%S%fZ.json")).write_text(content)
        target = reports / "predictions_latest.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(content)
        temporary.replace(target)
        print(f"Saved {len(predictions)} predictions for the next 24 hours.", flush=True)
        print(content, flush=True)

if __name__ == "__main__":
    main()
