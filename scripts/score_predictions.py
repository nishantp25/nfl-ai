"""Write prospective accuracy from archived pregame predictions."""
import json
from datetime import datetime, timezone
from pathlib import Path
from prediction_scoring import score
from nfldl.config import nfl_season_of
from nfldl.manifest import get_manifest
from nfldl.storage import read_dataset

ROOT = Path(__file__).resolve().parents[1]

def main():
    reports = ROOT / "reports"
    paths = sorted((reports / "predictions").glob("*.json"))
    snapshots = (json.loads(path.read_text()) for path in paths)
    games = read_dataset(
        get_manifest()["schedules"], seasons=list(range(2011, nfl_season_of() + 1))
    ).to_dicts()
    result = score(snapshots, games, datetime.now(timezone.utc))
    result["archive_files_checked"] = len(paths)
    result["limitation"] = "Measures archived predictions only; 12-hour result delay is a heuristic, and archive files are not tamper-proof."
    reports.mkdir(exist_ok=True)
    target = reports / "prediction_accuracy.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(target)
    summary = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
