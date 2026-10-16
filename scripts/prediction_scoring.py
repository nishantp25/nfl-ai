"""Score archived pregame predictions without counting repeated runs as games."""
from datetime import datetime, timedelta, timezone
import math

def score(snapshots, games, now):
    schedules = {g["game_id"]: g for g in games}
    selected = {}
    for snapshot in snapshots:
        generated = datetime.fromisoformat(snapshot["generated_at_utc"])
        if generated.utcoffset() is None:
            raise ValueError("Prediction timestamp must be timezone-aware")
        if generated > now:
            continue
        for prediction in snapshot["predictions"]:
            game = schedules.get(prediction["game_id"])
            if game is None:
                continue
            kickoff = game["kickoff_utc"]
            if kickoff is None or generated >= kickoff:
                continue
            home, away = game["home_score"], game["away_score"]
            if home is None or away is None or kickoff + timedelta(hours=12) > now:
                continue
            if home == away:
                continue
            probability = float(prediction["home_win_probability_conditional_on_no_tie"])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid prediction probability")
            previous = selected.get(game["game_id"])
            if previous is None or generated > previous[0]:
                selected[game["game_id"]] = (
                    generated, probability, int(home > away),
                    snapshot.get("model_directory"),
                )
    records = []
    for game_id, (generated, probability, outcome, model) in sorted(selected.items()):
        bounded = min(max(probability, 1e-15), 1 - 1e-15)
        records.append({
            "game_id": game_id, "prediction_generated_at_utc": generated.isoformat(),
            "model_directory": model, "home_win_probability": probability,
            "home_won": bool(outcome), "correct": bool((probability >= 0.5) == outcome),
            "brier_score": (probability - outcome) ** 2,
            "log_loss": -(outcome * math.log(bounded) + (1-outcome) * math.log(1-bounded)),
        })
    count = len(records)
    return {
        "checked_at_utc": now.astimezone(timezone.utc).isoformat(),
        "selection": "Latest archived prediction strictly before kickoff; one per decisive game",
        "games_scored": count,
        "accuracy": sum(r["correct"] for r in records) / count if count else None,
        "brier_score": sum(r["brier_score"] for r in records) / count if count else None,
        "log_loss": sum(r["log_loss"] for r in records) / count if count else None,
        "records": records,
    }
