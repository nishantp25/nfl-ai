"""Chronological NFL baseline; sports analytics only."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["POLARS_MAX_THREADS"] = "2"

import heapq
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nfldl.manifest import get_manifest
from nfldl.storage import read_dataset

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["elo_difference", "home_field", "rest_difference"]
ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

def build_rows(games, now):
    ratings, last_played, pending, rows = {}, {}, [], []
    for game in sorted(games, key=lambda g: (g["kickoff_utc"], g["game_id"])):
        kickoff = game["kickoff_utc"]
        cutoff = min(kickoff, now)
        while pending and pending[0][0] <= cutoff:
            _, _, home, away, home_score, away_score, played_at = heapq.heappop(pending)
            home_rating = ratings.get(home, 1500.0)
            away_rating = ratings.get(away, 1500.0)
            expected = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
            actual = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
            change = 20 * (actual - expected)
            ratings[home] = home_rating + change
            ratings[away] = away_rating - change
            last_played[home] = played_at
            last_played[away] = played_at
        home = ALIASES.get(game["home_team"], game["home_team"])
        away = ALIASES.get(game["away_team"], game["away_team"])
        def rest(team):
            previous = last_played.get(team)
            return min(max((kickoff - previous).total_seconds() / 86400, 0), 30) if previous else 7.0
        row = {
            "game_id": game["game_id"], "season": game["season"],
            "kickoff_utc": kickoff,
            "elo_difference": ratings.get(home, 1500.0) - ratings.get(away, 1500.0),
            "home_field": 0.0 if game.get("location") == "Neutral" else 1.0,
            "rest_difference": rest(home) - rest(away),
            "target": None,
        }
        hs, aws = game["home_score"], game["away_score"]
        available_at = kickoff + timedelta(hours=12)
        if hs is not None and aws is not None and available_at <= now:
            if hs != aws:
                row["target"] = int(hs > aws)
            heapq.heappush(pending, (available_at, game["game_id"], home, away, hs, aws, kickoff))
        rows.append(row)
    return rows

def matrix(rows):
    return np.asarray([[r[c] for c in FEATURES] for r in rows], dtype=float)

def evaluate(model, rows, constant):
    x = matrix(rows)
    y = np.asarray([r["target"] for r in rows], dtype=int)
    probabilities = model.predict_proba(x)[:, 1]
    reference = np.full(len(rows), constant)
    return {
        "games": len(rows),
        "accuracy": float(accuracy_score(y, probabilities >= 0.5)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "constant_home_win_log_loss": float(log_loss(y, reference, labels=[0, 1])),
    }

def main():
    now = datetime.now(timezone.utc)
    schedules = read_dataset(get_manifest()["schedules"], seasons=list(range(2011, now.year + 1)))
    schedules = schedules.filter(pl.col("kickoff_utc").is_not_null())
    rows = build_rows(schedules.to_dicts(), now)
    train = [r for r in rows if 2012 <= r["season"] <= 2023 and r["target"] is not None]
    assert len(train) > 1000, "Insufficient training history"
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    print(f"Training baseline on {len(train):,} games...", flush=True)
    model.fit(matrix(train), [r["target"] for r in train])
    constant = float(np.mean([r["target"] for r in train]))
    metrics = {}
    for season in (2024, 2025, now.year):
        sample = [r for r in rows if r["season"] == season and r["target"] is not None]
        if sample:
            metrics[str(season)] = evaluate(model, sample, constant)
    print(json.dumps(metrics, indent=2), flush=True)
    run = ROOT / "data/models" / now.strftime("baseline_%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    joblib.dump({"model": model, "features": FEATURES, "trained_through_season": 2023}, run / "model.joblib")
    upcoming = [r for r in rows if r["kickoff_utc"] > now]
    predictions = []
    if upcoming:
        for row, probability in zip(upcoming, model.predict_proba(matrix(upcoming))[:, 1]):
            predictions.append({
                "game_id": row["game_id"], "kickoff_utc": row["kickoff_utc"].isoformat(),
                "home_win_probability_conditional_on_no_tie": float(probability),
            })
    report = {
        "created_at_utc": now.isoformat(), "training_games": len(train),
        "training_seasons": [2012, 2023], "metrics": metrics,
        "limitations": [
            "Baseline candidate; no weather, injury, roster, or play-by-play features.",
            "Ties excluded from classification; probabilities conditional on a decisive result.",
            "Results become eligible 12 hours after kickoff; exact publication history unavailable.",
            "Historical schedules are current revisions, not archived pregame snapshots.",
            "No automatic production promotion.",
        ],
        "predictions": predictions,
    }
    (run / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved candidate model and report to {run}", flush=True)

if __name__ == "__main__":
    main()
