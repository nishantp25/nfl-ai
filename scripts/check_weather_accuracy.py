"""Compare saved pregame forecasts with reported postgame weather."""
import json
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
from nfldl.config import nfl_season_of
from nfldl.manifest import get_manifest
from nfldl.storage import read_dataset

ROOT = Path(__file__).resolve().parents[1]

def main():
    season = nfl_season_of()
    mf = get_manifest()
    forecasts = read_dataset(mf["weather_forecast"], seasons=[season])
    schedules = read_dataset(mf["schedules"], seasons=[season])
    completed = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
        & ~pl.col("roof").fill_null("unknown").str.to_lowercase().is_in(["dome", "closed"])
    ).select("game_id", pl.col("kickoff_utc").alias("actual_kickoff_utc"),
             ((pl.col("temp") - 32) * 5 / 9).alias("reported_temperature_c"),
             (pl.col("wind") * 1.609344).alias("reported_wind_kmh"))
    pairs = forecasts.filter(
        pl.col("weather_method") == "open_meteo_live_forecast"
    ).join(completed, on="game_id", how="inner").filter(
        pl.col("retrieved_at_utc") < pl.col("actual_kickoff_utc")
    ).with_columns(
        ((pl.col("actual_kickoff_utc") - pl.col("retrieved_at_utc"))
         .dt.total_seconds() / 3600).alias("lead_hours")
    ).with_columns(
        pl.when(pl.col("lead_hours") <= 2).then(pl.lit("0-2h"))
        .when(pl.col("lead_hours") <= 6).then(pl.lit("2-6h"))
        .when(pl.col("lead_hours") <= 24).then(pl.lit("6-24h"))
        .otherwise(pl.lit("over_24h")).alias("lead_bucket")
    ).sort("retrieved_at_utc").unique(
        subset=["game_id", "lead_bucket"], keep="last", maintain_order=True)
    metrics = []
    for bucket in ("0-2h", "2-6h", "6-24h", "over_24h"):
        frame = pairs.filter(pl.col("lead_bucket") == bucket)
        row = {"lead_bucket": bucket, "games": frame.height}
        for forecast, reported, label in (
            ("temperature_c", "reported_temperature_c", "temperature"),
            ("wind_speed_kmh", "reported_wind_kmh", "wind")):
            valid = frame.filter(pl.col(forecast).is_not_null() & pl.col(reported).is_not_null())
            row[label + "_comparisons"] = valid.height
            row[label + "_mae"] = (
                valid.select((pl.col(forecast) - pl.col(reported)).abs().mean()).item()
                if valid.height else None)
        metrics.append(row)
    payload = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": "Schedule-reported game weather; timing may differ from forecast hour.",
        "metrics": metrics,
    }
    target = ROOT / "reports/weather_accuracy.json"
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(target)
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
