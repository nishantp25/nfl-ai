"""Explore one-input sensitivity for nearby games; not calibrated uncertainty."""
import os
os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import fcntl
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib
import numpy as np
import polars as pl
from train_live import build_rows, FEATURES
from performance_features import enrich
from parameter_sensitivity import sweep
from nfldl.manifest import get_manifest
from nfldl.storage import read_dataset

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ["elo_difference", "off_epa_prior5_diff", "off_success_prior5_diff",
             "pass_rate_prior5_diff", "def_epa_allowed_prior5_diff",
             "def_success_allowed_prior5_diff"]

def main():
    now = datetime.now(timezone.utc)
    reports = ROOT / "reports"
    with (reports / "sensitivity.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Sensitivity analysis already running; skipped.", flush=True)
            return
        snapshot = json.loads((reports / "predictions_latest.json").read_text())
        generated = datetime.fromisoformat(snapshot["generated_at_utc"])
        if not 0 <= (now - generated).total_seconds() <= 3600:
            raise RuntimeError("Refresh predictions first; snapshot is over one hour old.")
        predictions = [r for r in snapshot["predictions"]
                       if now < datetime.fromisoformat(r["kickoff_utc"]) <= now + timedelta(hours=24)]
        results = []
        if predictions:
            saved = joblib.load(Path(snapshot["model_directory"]) / "model.joblib")
            assert saved["features"] == FEATURES
            schedules = read_dataset(get_manifest()["schedules"], seasons=list(range(2011, 2024)))
            schedules = schedules.filter(pl.col("kickoff_utc").is_not_null()).to_dicts()
            performance = []
            for season in range(2011, 2024):
                path = ROOT / "data/curated/team_performance" / f"season={season}/part.parquet"
                performance.extend(pl.read_parquet(path).to_dicts())
            historical = enrich(build_rows(schedules, now), performance, schedules, now)
            training = [r for r in historical if r["season"] >= 2012 and r["target"] is not None]
            for prediction in predictions:
                base = [np.nan if prediction["feature_values"][f] is None
                        else prediction["feature_values"][f] for f in FEATURES]
                changes = {}
                for name in VARIABLES:
                    values = np.asarray([r[name] for r in training], dtype=float)
                    values = values[np.isfinite(values)]
                    low, high = np.quantile(values, [0.05, 0.95])
                    changes[name] = sweep(saved["model"], base, FEATURES.index(name), low, high)
                    print(f"{prediction['game_id']}: completed {name} sweep", flush=True)
                results.append({
                    "game_id": prediction["game_id"],
                    "base_home_win_probability": prediction["home_win_probability_conditional_on_no_tie"],
                    "sweeps": changes,
                })
        payload = {
            "generated_at_utc": now.isoformat(),
            "prediction_snapshot_utc": snapshot["generated_at_utc"],
            "model_directory": snapshot["model_directory"],
            "bounds": "5th-95th percentiles of 2012-2023 training features",
            "limitations": "Marginal scenarios can be jointly unrealistic. Ranges and averages are not confidence intervals or improved forecasts. Weather and injuries are not modeled.",
            "results": results,
        }
        target = reports / "sensitivity_latest.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
        temporary.replace(target)
        print(f"Saved sensitivity results for {len(results)} games.", flush=True)

if __name__ == "__main__":
    main()
