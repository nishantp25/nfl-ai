"""Pregame performance features with conservative result-availability cutoffs."""
from collections import defaultdict, deque
from datetime import timedelta
import math

ALIASES = {"STL": "LA", "SD": "LAC", "OAK": "LV"}
MEASURES = ["off_epa", "off_success", "pass_rate",
            "def_epa_allowed", "def_success_allowed"]
FEATURES = [name + "_prior5_diff" for name in MEASURES] + [
    "home_performance_games", "away_performance_games"]

def enrich(rows, performance, schedules, now):
    games = {g["game_id"]: g for g in schedules}
    events = []
    for record in performance:
        game = games.get(record["game_id"])
        if game is None:
            raise ValueError("Performance record has no schedule")
        if game["home_score"] is None or game["away_score"] is None:
            continue
        available = game["kickoff_utc"] + timedelta(hours=12)
        if available > now:
            continue
        team = ALIASES.get(record["team"], record["team"])
        events.append((available, record["game_id"], team, record))
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    history = defaultdict(lambda: deque(maxlen=5))
    output = []
    index = 0
    for original in sorted(rows, key=lambda r: (r["kickoff_utc"], r["game_id"])):
        cutoff = min(original["kickoff_utc"], now)
        while index < len(events) and events[index][0] < cutoff:
            available, game_id, team, record = events[index]
            history[team].append((available, game_id, record))
            index += 1
        game = games[original["game_id"]]
        home = ALIASES.get(game["home_team"], game["home_team"])
        away = ALIASES.get(game["away_team"], game["away_team"])
        for team in (home, away):
            assert all(t < cutoff and gid != original["game_id"]
                       for t, gid, _ in history[team]), "Future or own-game data"
        row = dict(original)
        row["home_performance_games"] = len(history[home])
        row["away_performance_games"] = len(history[away])
        def mean(team, measure):
            values = [r[measure] for _, _, r in history[team]
                      if r.get(measure) is not None and math.isfinite(r[measure])]
            return sum(values) / len(values) if values else float("nan")
        for measure in MEASURES:
            row[measure + "_prior5_diff"] = mean(home, measure) - mean(away, measure)
        output.append(row)
    return output
