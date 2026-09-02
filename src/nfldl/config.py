"""Runtime configuration for the NFL data layer.

Everything that is environment-specific lives here. Nothing in this module
encodes *data* facts (season coverage, schemas, providers) -- those belong in
``configs/data_sources.yaml`` and are verified at runtime by ``nfldl.verify``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = _env_path("NFLDL_PROJECT_ROOT", PACKAGE_ROOT.parents[1])

CONFIG_DIR = _env_path("NFLDL_CONFIG_DIR", PROJECT_ROOT / "configs")
DATA_DIR = _env_path("NFLDL_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
REPORT_DIR = _env_path("NFLDL_REPORT_DIR", PROJECT_ROOT / "reports")
CACHE_DIR = _env_path("NFLDL_CACHE_DIR", DATA_DIR / "_cache")

MANIFEST_PATH = CONFIG_DIR / "data_sources.yaml"
COVERAGE_REPORT_PATH = REPORT_DIR / "verified_coverage.json"
QUALITY_REPORT_PATH = REPORT_DIR / "quality_report.json"
PROVENANCE_LOG_PATH = REPORT_DIR / "ingestion_log.jsonl"


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    """Process-wide settings. Override via environment variables."""

    # Modelling window requested by the project brief. This is an *aspiration*,
    # not a claim: `verify` discovers what actually exists and `ingest` clips to
    # the intersection. Never treat this as documented coverage.
    target_season_start: int = int(os.environ.get("NFLDL_SEASON_START", 2011))
    target_season_end: int = int(os.environ.get("NFLDL_SEASON_END", 2026))

    http_timeout: int = int(os.environ.get("NFLDL_HTTP_TIMEOUT", 60))
    http_retries: int = int(os.environ.get("NFLDL_HTTP_RETRIES", 4))
    http_backoff: float = float(os.environ.get("NFLDL_HTTP_BACKOFF", 1.5))
    max_workers: int = int(os.environ.get("NFLDL_MAX_WORKERS", 8))

    user_agent: str = os.environ.get(
        "NFLDL_USER_AGENT",
        "nfl-data-layer/1.0 (research; contact: set NFLDL_USER_AGENT)",
    )

    # Weather provider. "open_meteo_archive" (ERA5, no key) is the default.
    # "meteostat" requires `pip install meteostat`. "none" disables weather.
    weather_provider: str = os.environ.get("NFLDL_WEATHER_PROVIDER", "open_meteo_archive")

    # Refuse to write a partition whose season is not in verified coverage.
    strict_coverage: bool = os.environ.get("NFLDL_STRICT_COVERAGE", "1") == "1"

    directories: tuple[Path, ...] = field(
        default=(RAW_DIR, INTERIM_DIR, REPORT_DIR, CACHE_DIR), repr=False
    )

    @property
    def target_seasons(self) -> list[int]:
        return list(range(self.target_season_start, self.target_season_end + 1))

    def ensure_dirs(self) -> None:
        for d in self.directories:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


# --------------------------------------------------------------------------
# Season helpers
#
# We deliberately do NOT depend on nflreadpy.get_current_season() for control
# flow, because it is a date heuristic and it raises on future seasons. We use
# it only as a cross-check and log disagreements.
# --------------------------------------------------------------------------

def nfl_season_of(day: date | None = None) -> int:
    """Season label for a calendar date.

    A season is labelled by the calendar year it starts in. The league year
    rolls over in March; the playing season starts the Thursday after Labor Day.
    Anything before September belongs to the previously-played season.
    """
    day = day or date.today()
    # First Monday of September = Labor Day; season opens the Thursday after.
    labor_day = next(
        date(day.year, 9, d) for d in range(1, 8) if date(day.year, 9, d).weekday() == 0
    )
    opener = date(labor_day.year, 9, labor_day.day + 3)
    return day.year if day >= opener else day.year - 1


def league_year_of(day: date | None = None) -> int:
    """Roster/league year (rolls over mid-March, when free agency opens)."""
    day = day or date.today()
    return day.year if day >= date(day.year, 3, 15) else day.year - 1
