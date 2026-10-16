# Local NFL analytics host

Project location: ~/Documents/nfl-ai on the internal drive.
Python: .venv/bin/python (Python 3.12).

## Unattended operation

Keep the Mac plugged in, awake, and the user logged in. Closed-lid operation is supported with a connected external display, external keyboard/mouse, and power.
LaunchAgents resume after login; they do not run while logged out.
A locked screen is compatible with a logged-in session.

## Schedules

- com.nishant.nfl-ai.weather: every 3600 seconds.
- com.nishant.nfl-ai.core: even-numbered hours at :15, local time.
- com.nishant.nfl-ai.predictions: every hour at :05 and :35.
- com.nishant.nfl-ai.sensitivity: every hour at :10 and :40.

Definitions are in ~/Library/LaunchAgents/.
Logs and status reports are in reports/.
Weather, core refreshes, and prediction reads share a nonblocking lock.
A conflicting run skips and waits for its next scheduled opportunity.
Sensitivity uses its own lock and batches of 8192 rows.

## Outputs

- reports/core_refresh.json: last successful core cycle.
- reports/weather_latest.json: latest weather comparison report.
- reports/weather_accuracy.json: available postgame weather comparisons.
- reports/predictions_latest.json: predictions for kickoff within 24 hours.
- reports/predictions/: timestamped prediction snapshots.
- reports/sensitivity_latest.json: latest one-input sensitivity results.
- data/models/live_*/: saved live candidates and evaluation reports.
- data/quarantine/player_stats/: player-stat rows without player IDs.

An empty prediction list is expected when no games start within 24 hours.
Check report timestamps as well as job exit codes.
Failed runs can leave an older successful report in place.
Logs, model artifacts, and snapshot data currently have no retention policy.

## Model scope

The live model uses Elo, home-field context, rest, and prior-five-game
play-by-play aggregates. Weights are fitted on eligible completed games
since 2012. Historical evaluation uses a separate fit through 2023.
Displayed historical metrics do not measure the refitted live model's
future accuracy. Ties are excluded from classification.

Weather, injuries, and roster information are collected but are not model
inputs yet. Hourly weather checks use delayed schedule-reported game
weather where available; they are not live station verification.

Result and performance availability uses a conservative 12-hour delay
after kickoff, not a verified historical publication timestamp.
Historical source files contain revisions. Backtests are retrospective,
not a complete reconstruction of what was published before each game.

Sensitivity runs evaluate six inputs separately across 1,000,001 grid
values each. Bounds are development-data marginal percentiles.
Other inputs remain fixed. These are not joint outcome simulations,
confidence intervals, or an accuracy improvement.

## Recreating the environment

Install Python 3.12, create .venv, install requirements-macos-arm64.lock.txt,
then install the project in editable mode. Restore the LaunchAgent
definitions separately; their paths are specific to this Mac.

## Git checkpoints

Ask the user for the actual date/time before any commit or push checkpoint.
Do not fabricate commit history. Local data and reports are ignored by Git.

## Injury observations and prospective accuracy

The core refresh collects and validates append-only injury_snapshots.
retrieved_at_utc records when the local collector observed the source;
it does not establish the source's original publication time.
These observations are collected but are not model inputs yet.

The core refresh also writes reports/prediction_accuracy.json.
It selects the latest archived prediction strictly before actual kickoff,
counts each decisive game once, and skips ties and unavailable results.
Results become eligible 12 hours after kickoff as a conservative heuristic.
Accuracy remains null until eligible archived predictions can be scored.
