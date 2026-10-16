from datetime import datetime, timedelta, timezone
from pathlib import Path
import runpy
import pytest

score = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prediction_scoring.py"))["score"]
KICKOFF = datetime(2026, 10, 18, 17, tzinfo=timezone.utc)

def game(home=24, away=17):
    return {"game_id": "example", "kickoff_utc": KICKOFF,
            "home_score": home, "away_score": away}

def snapshot(hours_before, probability):
    return {
        "generated_at_utc": (KICKOFF - timedelta(hours=hours_before)).isoformat(),
        "model_directory": "example-model",
        "predictions": [{"game_id": "example",
                         "home_win_probability_conditional_on_no_tie": probability}],
    }

def test_latest_pregame_prediction_counts_once():
    result = score([snapshot(3, 0.2), snapshot(1, 0.8)], [game()], KICKOFF + timedelta(days=1))
    assert result["games_scored"] == 1
    assert result["accuracy"] == 1
    assert result["brier_score"] == pytest.approx(0.04)
    assert result["records"][0]["home_win_probability"] == 0.8

def test_exact_kickoff_and_postgame_predictions_are_excluded():
    result = score([snapshot(0, 0.9), snapshot(-1, 0.9)], [game()], KICKOFF + timedelta(days=1))
    assert result["games_scored"] == 0
    assert result["accuracy"] is None

@pytest.mark.parametrize("home,away", [(17,17), (None,None)])
def test_ties_and_unfinished_games_are_not_scored(home, away):
    result = score([snapshot(1, 0.8)], [game(home,away)], KICKOFF + timedelta(days=1))
    assert result["games_scored"] == 0

def test_results_wait_for_availability_cutoff():
    result = score([snapshot(1, 0.8)], [game()], KICKOFF + timedelta(hours=2))
    assert result["games_scored"] == 0

@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_invalid_probabilities_rejected(probability):
    with pytest.raises(ValueError, match="probability"):
        score([snapshot(1, probability)], [game()], KICKOFF + timedelta(days=1))

def test_later_forecast_cannot_overwrite_pregame_record():
    result = score([snapshot(1, 0.2), snapshot(-2, 0.9)], [game()], KICKOFF + timedelta(days=1))
    assert result["games_scored"] == 1
    assert result["accuracy"] == 0
    assert result["records"][0]["home_win_probability"] == 0.2
