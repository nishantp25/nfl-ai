# nfldl — a manifest-driven, leakage-aware NFL data layer

Raw data acquisition for NFL game forecasting. This package does one job and
tries to do it without lying to you: it gets canonical data onto disk, records
exactly where every row came from, and makes it hard to accidentally build a
model that already knows the answer.

It deliberately does **not** compute Elo, opponent-adjusted EPA, strength of
schedule, or any of the other derived features. Those are downstream, and the
manifest actively forbids downloading them from third parties — see
[Derived features](#derived-features-are-forbidden-downloads) below.

---

## Quick start

```bash
pip install -e ".[dev]"

nfldl verify                      # probe what actually exists, before trusting anything
nfldl ingest --required-only      # fetch the required datasets for the target window
nfldl validate                    # quality checks against what landed on disk
nfldl doctor                      # one-screen health summary
```

Then look at `examples/pregame_frame.py` for the intended downstream shape.

---

## The five ideas

### 1. The manifest is the contract

`configs/data_sources.yaml` declares every dataset: provider, access path,
historical coverage, update frequency, storage path, primary key, timestamp
field, licence notes, and whether it is required. Nothing else is downloadable.

`registry.py` asserts at **import time** that the set of registered fetchers is
exactly the set of declared datasets. Adding a convenient dataset without
documenting it is a hard failure, not a code review comment.

```bash
nfldl manifest        # the declared source table
nfldl manifest --json
```

### 2. Coverage is verified, never assumed

Your brief said: *do not assume every dataset covers 2011–2026.* That is
enforced mechanically. `nfldl verify` HEAD-probes the live provider — no
payloads downloaded — and writes `reports/verified_coverage.json`, which the
manifest overlays on load. `ingest` then clips to verified coverage and reports
skipped seasons explicitly:

```
participation      ingested=1, skipped_unavailable=2   rows=48,434
2 partition(s) skipped as unavailable -- this is coverage, not error.
```

Verified against the live nflverse releases on 2026-09-02:

| Dataset | Real coverage | Gap vs. the 2011–2026 target |
|---|---|---|
| `schedules` | 1999–2026 | — |
| `rosters`, `weekly_rosters`, `depth_charts` | → 2026 | — (league-year assets publish early) |
| `play_by_play` | 1999–2025 | 2026 |
| `injuries` | 2009–2025 | 2026 |
| `player_stats`, `team_stats` | 1999–2025 | 2026 |
| `snap_counts` | 2012–2025 | 2011, 2026 |
| `participation` | 2016–2025 | **2011–2015, 2026** |
| `pfr_advstats` | 2018–2025 | **2011–2017, 2026** |
| `ftn_charting` | 2022–2025 | **2011–2021, 2026** |
| `nextgen_stats` | 2016– | **2011–2015** |

Re-run `nfldl verify --write-manifest` after any nflverse release change; it
rewrites only the `verified_*` fields and leaves the comments intact.

> **The September rollover.** `nflreadpy.get_current_season()` is a date
> heuristic and *raises* on seasons it considers future — including ones whose
> assets genuinely exist (2026 rosters, right now). `sources/nflverse.py` catches
> that specific `ValueError`, confirms the asset is really there with a HEAD
> request, and reads the identical release file directly. Same provider, same
> bytes; we route around a client-side date check, not around the source.

### 3. Point-in-time discipline

`pit.py` is the module that stops the future leaking backwards.

| Primitive | What it does |
|---|---|
| `knowledge_time(df, dataset)` | attaches `_known_at` — when the row became knowable |
| `asof_join(left, right, by=…)` | strictly-backward join; **exact ties are excluded** |
| `lag_weekly(df, cols, weeks=1)` | shifts postgame weekly measures forward |
| `audit_frame(df, dataset_of=…)` | scans a feature frame and raises on leaks |
| `assert_no_future_rows(df)` | hard check after any join |

Datasets are classified `pregame` / `postgame` / `static` / `unknown`, and
`unknown` is treated as postgame — unclear provenance is not a licence.
`play_by_play`, `player_stats`, `team_stats`, `snap_counts`, `participation`,
`ftn_charting`, `nextgen_stats` and `pfr_advstats` are all **postgame**: the
week-W row exists only because week W was played.

`audit_frame` catches the ordinary mistakes — an outcome column left in the
feature set, an unlagged postgame measure, betting-market columns used without
saying so, and `_ingested_at` treated as a feature (processing time is not
knowledge time).

The example verifies the property end to end:

```
week 1 of 2021, form should be all null: True
week 5 of 2021, form should be mostly populated: 1.0
assert_no_future_rows: passed
audit correctly rejected the leaky variant:
  - result: outcome column from the game being predicted
```

### 4. Every row is attributable

Each partition is written atomically (temp file + rename, so an interrupted run
can never leave a half-written file a later run mistakes for complete) with
three stamped columns and a JSON sidecar carrying provider, tier, access path,
row/column counts, a SHA-256 of the content, and the manifest version in force.
Everything is also appended to `reports/ingestion_log.jsonl`.

```bash
nfldl inventory
```

### 5. Conflicts resolve down a documented ladder

`conflicts.py` implements your priority order — official NFL → nflverse →
established databases → other reputable public → curated fallback. Departing
from it requires an `Override` carrying a reason, a source URL and an author;
the dataclass refuses to construct without them. Disagreements are logged to
`reports/reconciliations.jsonl` rather than silently smoothed over.

---

## Derived features are forbidden downloads

The manifest lists 23 features that must be computed from canonical raw data:
Elo, Bayesian team ratings, offensive/defensive strength, QB and player-value
ratings, injury impact, roster and O-line continuity, receiving-corps and
defensive-front strength, recent form, EWMA performance, opponent-adjusted EPA,
strength of schedule, head-to-head, coaching and scheme tendencies, matchup
advantages, rest, travel, and feature differentials.

Registering a fetcher for any of them raises `ManifestError`. This is what makes
historical reconstruction possible: a rating you computed from plays that had
already happened can be rebuilt for any past week; a rating you downloaded today
cannot.

---

## Sources, and why

**nflverse via `nflreadpy`** for everything it provides — play-by-play,
schedules, players, rosters (season and weekly), depth charts, injuries, snap
counts, player/team stats, participation, FTN charting, NGS, draft picks,
trades, contracts, officials, combine, and the PFR advanced-stats mirror.
Play-by-play is the foundation; EPA, success rate, efficiency, explosive rate,
sack rate, turnovers, drive and situational performance and all tendency
features are derived from it rather than fetched separately.

**Schedules is the canonical game table.** Head-to-head, rest, travel, roof and
surface context all come from it. There is no separate H2H file. `kickoff_utc`
is derived once, in `sources/nflverse.py`, so three modules cannot disagree
about timezones. (nflverse stamps `gametime` in US/Eastern even for London,
Munich, Mexico City and São Paulo games, so a single Eastern interpretation is
correct throughout.)

**Coaching is derived, not curated.** `home_coach`/`away_coach` are on every
schedule row, so head-coach tenure — including mid-season firings, dated to the
exact game, and a returning coach correctly split into two stints — falls out of
run-length encoding the canonical game table. Only coordinators need curation;
`configs/coaching_staff.template.csv` is the starting point, and rows without a
`source_url` are rejected. Absent the file, the layer warns loudly and produces
head coaches only — coordinator features are *unavailable*, not silently wrong.

Coach identity is name-slugged, which your brief rightly warns against. There is
no coach ID anywhere in the public data, so it is unavoidable; it is confined to
this one dimension and never used to join player data. Player joins always use
stable IDs.

**Stadiums** are SCD-2 (`valid_from` / `valid_to`), because teams relocate and
roofs change — a 2015 Rams home game resolves to St. Louis, not Los Angeles.
Per-game `roof` and `surface` come from schedules, which already handles roof
state and re-turfing correctly; the curated CSV supplies only what schedules
lacks (coordinates, elevation, timezone). All 52 venues that appear in schedules
from 2011 on are covered, including the international sites.

**Weather** uses the Open-Meteo Historical Weather API (ERA5) — free, keyless,
global, 1940-present, so there is no coverage gap anywhere in the window. NOAA
and Meteostat are documented alternatives; NOAA needs a token and is US-only.
One request per venue-season rather than per game.

---

## Two caveats I want to be explicit about

**Weather is reanalysis, not forecast.** ERA5 is what the weather *actually
was*, not what anyone knew beforehand. For a pregame feature this is mildly
optimistic. Every row carries `weather_is_reanalysis = true`, and `pit` treats
the dataset as `unknown` (i.e. postgame-adjacent) unless you opt in. If that
optimism matters for your use case, you need archived forecasts, not ERA5.

**Participation changed hands after 2023.** The upstream feed moved from NGS to
FTN, so column fill rates are not stable across the 2023/2024 boundary. Check
per-season null rates before modelling — `nfldl validate` already flags them.
There is nothing before 2016; do not impute personnel or formation for
2011–2015, because those features simply do not exist there.

---

## Testing

```bash
python -m pytest tests -q      # 31 tests
```

The suite covers the invariants that matter: manifest/registry agreement,
coverage clipping rather than extrapolation, as-of joins excluding exact ties,
the audit catching outcome columns and unlagged postgame measures, mid-season
coaching changes producing two stints, overrides requiring documentation, and
the quality checks catching duplicate keys, wrong manifest keys, missing
seasons, float-typed season columns and near-empty columns.

One test — `test_high_leakage_datasets_are_marked_postgame` — exists because it
caught a real inconsistency during development: `injuries` had been tagged
`leakage_risk: high`, but the table is genuinely pregame (reports land Wed–Sat).
The risk is in the *join*, not the data. The manifest now says `medium` with an
explicit `leakage_note`, rather than mislabelling the source.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `NFLDL_SEASON_START` / `NFLDL_SEASON_END` | 2011 / 2026 | target window (an aspiration, never a coverage claim) |
| `NFLDL_DATA_DIR` | `./data` | lake root |
| `NFLDL_STRICT_COVERAGE` | `1` | refuse to ingest without a coverage report |
| `NFLDL_WEATHER_PROVIDER` | `open_meteo_archive` | or `meteostat`, or `none` |
| `NFLDL_USER_AGENT` | — | **set this** to something with a contact address |
| `NFLDL_MAX_WORKERS` | 8 | probe concurrency |

---

## Licensing

nflverse data is CC-BY 4.0, except FTN charting which is CC-BY-SA 4.0
(share-alike — stricter than the rest). PFR advanced stats are consumed through
the nflverse mirror only; do not scrape pro-football-reference.com directly or
circumvent access controls. Open-Meteo is free for non-commercial use with
attribution; ERA5 is Copernicus/ECMWF licensed. Check current terms before any
commercial use.

---

## What is deliberately not here

Feature engineering, model training, backtesting, and serving. The boundary is
intentional: this layer's job is to make the raw material trustworthy and its
provenance legible. `examples/pregame_frame.py` shows the handoff.

Big Data Bowl tracking data is also absent by design. Your brief called it
optional, and it does not exist for every game — treat it as an enhancement to
evaluate against out-of-sample forecasts, not a baseline input.
