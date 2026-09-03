"""Tests for the parts that must not silently break.

The emphasis is on the invariants the brief cares about: the manifest is
authoritative, coverage is never assumed, and nothing from after kickoff can
reach a pregame frame.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import polars as pl
import pytest

from nfldl import conflicts, pit, quality
from nfldl.manifest import Coverage, ManifestError, get_manifest, load_manifest
from nfldl.registry import REGISTRY, resolve_order
from nfldl.sources import coaching

UTC = timezone.utc


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def test_manifest_loads_and_every_dataset_has_a_fetcher():
    mf = get_manifest()
    assert set(mf.names()) == set(REGISTRY), (
        "manifest and registry have drifted; that is exactly the failure the "
        "registry validation exists to prevent"
    )


def test_required_datasets_declare_a_primary_key():
    mf = get_manifest()
    for name in mf.required_names():
        assert mf[name].primary_key, f"{name} is required but keyless"


def test_derived_features_cannot_be_registered():
    mf = get_manifest()
    assert "elo_ratings" in mf.forbidden_downloads
    with pytest.raises(ManifestError, match="DERIVED"):
        mf.assert_not_forbidden("opponent_adjusted_epa")


def test_every_dataset_has_a_documented_role_and_license_position():
    mf = get_manifest()
    for name, spec in mf.datasets.items():
        assert spec.role.strip(), f"{name} has no documented role"
        assert spec.provider.strip(), f"{name} has no provider"
        assert mf.tier_rank(spec.tier) >= 1


def test_high_leakage_datasets_are_marked_postgame():
    """A dataset flagged high-risk in the manifest must be handled as postgame."""
    mf = get_manifest()
    for name, spec in mf.datasets.items():
        if spec.leakage_risk == "high":
            assert pit.DATASET_AVAILABILITY.get(name) in ("postgame", "unknown"), (
                f"{name} is leakage_risk=high but pit treats it as "
                f"{pit.DATASET_AVAILABILITY.get(name)}"
            )


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------

def test_coverage_clips_rather_than_extrapolates():
    cov = Coverage(2016, 2025)
    assert cov.clip(range(2011, 2027)) == list(range(2016, 2026))
    assert 2011 not in cov.seasons()


def test_open_ended_coverage_resolves_against_current_season():
    cov = Coverage(2020, None)
    assert cov.resolve_end(current_season=2024) == 2024
    assert cov.seasons(current_season=2022) == [2020, 2021, 2022]


def test_participation_manifest_does_not_claim_the_full_target_window():
    """The brief's headline trap: participation has nothing before 2016."""
    spec = get_manifest()["participation"]
    assert spec.declared.start == 2016
    assert 2011 not in spec.verified.seasons(current_season=2025)


def test_unsampled_coverage_report_does_not_override_manifest(tmp_path):
    """HEAD reachability alone is not evidence for observed runtime coverage."""
    manifest = load_manifest()
    original = manifest["participation"].verified

    report_path = tmp_path / "coverage.json"
    report_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "participation": {
                        "verified_start": 2020,
                        "verified_end": 2021,
                        "schema_verified": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    overlaid = manifest.with_verified_coverage(report_path)

    assert overlaid["participation"].verified == original


def test_schema_verified_coverage_report_can_override_manifest(tmp_path):
    """Observed, schema-verified coverage may replace advisory manifest coverage."""
    manifest = load_manifest()

    report_path = tmp_path / "coverage.json"
    report_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "participation": {
                        "verified_start": 2018,
                        "verified_end": 2024,
                        "schema_verified": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    overlaid = manifest.with_verified_coverage(report_path)

    assert overlaid["participation"].verified == Coverage(2018, 2024)


def test_dependency_order_puts_inputs_first():
    order = resolve_order(["weather", "coaching_staff"])
    assert order.index("schedules") < order.index("weather")
    assert order.index("stadiums") < order.index("weather")


# --------------------------------------------------------------------------
# Point in time
# --------------------------------------------------------------------------

def _kickoffs() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "team": ["KC", "KC"],
            "kickoff_utc": [
                datetime(2023, 9, 10, 17, 0, tzinfo=UTC),
                datetime(2023, 9, 17, 17, 0, tzinfo=UTC),
            ],
        }
    )


def test_postgame_knowledge_time_is_after_kickoff():
    df = pit.knowledge_time(_kickoffs(), "snap_counts")
    assert (df.get_column("_known_at") > df.get_column("kickoff_utc")).all()


def test_pregame_knowledge_time_is_not_after_kickoff():
    df = pit.knowledge_time(_kickoffs(), "injuries")
    assert (df.get_column("_known_at") <= df.get_column("kickoff_utc")).all()


def test_asof_join_excludes_a_games_own_postgame_row():
    games = _kickoffs()
    postgame = pit.knowledge_time(
        games.with_columns(pl.Series("snaps", [70, 65])), "snap_counts"
    )
    joined = pit.asof_join(
        games,
        postgame.select("team", "snaps", "_known_at"),
        by=["team"],
    )

    # Week 1 has no prior game, so nothing may be attached.
    wk1 = joined.filter(pl.col("game_id") == "g1")
    assert wk1.get_column("snaps").is_null().all()

    # Week 2 may see week 1 only.
    wk2 = joined.filter(pl.col("game_id") == "g2")
    assert wk2.get_column("snaps").to_list() == [70]


def test_asof_join_rejects_exact_ties():
    """A row knowable exactly at kickoff is not knowable *before* kickoff."""
    games = _kickoffs().head(1)

    same_instant = (
        games.select("team", "kickoff_utc")
        .with_columns(
            pl.col("kickoff_utc").alias("_known_at"),
            pl.lit(1).alias("flag"),
        )
        .drop("kickoff_utc")
    )

    joined = pit.asof_join(
        games,
        same_instant,
        by=["team"],
    )

    assert joined.get_column("flag").is_null().all()


def test_audit_frame_catches_the_outcome_column():
    frame = pl.DataFrame(
        {
            "game_id": ["g1"],
            "result": [7],
            "elo_diff": [12.0],
        }
    )

    with pytest.raises(pit.LeakageError, match="result"):
        pit.audit_frame(frame)


def test_audit_frame_accepts_a_lagged_postgame_feature():
    frame = pl.DataFrame(
        {
            "game_id": ["g1"],
            "off_snaps_lag1": [70],
        }
    )

    findings = pit.audit_frame(
        frame,
        dataset_of={"off_snaps_lag1": "snap_counts"},
    )

    assert not [
        finding
        for finding in findings
        if finding.severity == "error"
    ]


def test_audit_frame_errors_on_an_unlagged_postgame_feature():
    frame = pl.DataFrame(
        {
            "game_id": ["g1"],
            "off_snaps": [70],
        }
    )

    with pytest.raises(pit.LeakageError, match="off_snaps"):
        pit.audit_frame(
            frame,
            dataset_of={"off_snaps": "snap_counts"},
        )


def test_market_columns_warn_but_do_not_block():
    frame = pl.DataFrame(
        {
            "spread_line": [-3.0],
        }
    )

    findings = pit.audit_frame(frame)

    assert [
        finding.severity
        for finding in findings
    ] == ["warning"]


def test_assert_no_future_rows():
    df = pl.DataFrame(
        {
            "kickoff_utc": [
                datetime(
                    2023,
                    9,
                    10,
                    17,
                    tzinfo=UTC,
                )
            ],
            "_known_at": [
                datetime(
                    2023,
                    9,
                    10,
                    21,
                    tzinfo=UTC,
                )
            ],
        }
    )

    with pytest.raises(pit.LeakageError):
        pit.assert_no_future_rows(df)


def test_lag_weekly_shifts_forward():
    df = pl.DataFrame(
        {
            "season": [2023] * 3,
            "team": ["KC"] * 3,
            "week": [1, 2, 3],
            "epa": [0.1, 0.2, 0.3],
        }
    )

    out = pit.lag_weekly(
        df,
        ["epa"],
    ).sort("week")

    assert out.get_column("epa_lag1").to_list() == [
        None,
        0.1,
        0.2,
    ]


# --------------------------------------------------------------------------
# Conflict resolution
# --------------------------------------------------------------------------

def test_priority_ladder_prefers_the_higher_tier():
    res = conflicts.resolve(
        "game:surface",
        [
            conflicts.Candidate(
                "turf",
                "curated_fallback",
                "stadiums",
            ),
            conflicts.Candidate(
                "grass",
                "official_nfl",
                "schedules",
            ),
        ],
        log_to_file=False,
    )

    assert res.chosen.value == "grass"
    assert res.disagreement


def test_override_requires_documentation():
    with pytest.raises(ValueError, match="reason"):
        conflicts.Override(
            "k",
            "stadiums",
            "",
            "https://x",
            "me",
            "2026-01-01",
        )


def test_documented_override_beats_the_ladder():
    ov = conflicts.Override(
        key="game:surface",
        prefer_dataset="stadiums",
        reason="schedules mislabels the 2015 temporary field",
        source_url="https://example.org/notice",
        recorded_by="analyst",
        recorded_on="2026-01-01",
    )

    res = conflicts.resolve(
        "game:surface",
        [
            conflicts.Candidate(
                "grass",
                "official_nfl",
                "schedules",
            ),
            conflicts.Candidate(
                "turf",
                "curated_fallback",
                "stadiums",
            ),
        ],
        overrides={
            "game:surface": ov,
        },
        log_to_file=False,
    )

    assert res.chosen.value == "turf"
    assert res.rule == "documented_override"


def test_coalesce_by_priority_flags_disagreement():
    df = pl.DataFrame(
        {
            "official_roof": [
                "closed",
                None,
            ],
            "curated_roof": [
                "dome",
                "dome",
            ],
        }
    )

    out = conflicts.coalesce_by_priority(
        df,
        "roof",
        {
            "official_roof": "official_nfl",
            "curated_roof": "curated_fallback",
        },
    )

    assert out.get_column("roof").to_list() == [
        "closed",
        "dome",
    ]

    assert out.get_column("roof_disagreement").to_list() == [
        True,
        False,
    ]

    assert out.get_column("roof_source_tier").to_list() == [
        "official_nfl",
        "curated_fallback",
    ]


# --------------------------------------------------------------------------
# Coaching derivation
# --------------------------------------------------------------------------

def test_head_coach_stints_split_on_a_midseason_change():
    sched = pl.DataFrame(
        {
            "season": [2022] * 4,
            "game_id": [
                "a",
                "b",
                "c",
                "d",
            ],
            "gameday": [
                "2022-09-11",
                "2022-09-18",
                "2022-10-02",
                "2022-10-09",
            ],
            "home_team": ["CAR"] * 4,
            "away_team": ["ATL"] * 4,
            "home_coach": [
                "Matt Rhule",
                "Matt Rhule",
                "Steve Wilks",
                "Steve Wilks",
            ],
            "away_coach": [
                "Arthur Smith",
            ] * 4,
        }
    )

    stints = coaching.head_coach_stints(
        sched
    ).filter(
        pl.col("team") == "CAR"
    )

    assert stints.height == 2

    assert stints.get_column("coach_name").to_list() == [
        "Matt Rhule",
        "Steve Wilks",
    ]

    assert (
        str(
            stints.get_column("end_date").to_list()[0]
        )
        == "2022-09-18"
    )


def test_head_coach_stints_separate_a_return_after_a_gap():
    sched = pl.DataFrame(
        {
            "season": [
                2020,
                2021,
                2022,
            ],
            "game_id": [
                "a",
                "b",
                "c",
            ],
            "gameday": [
                "2020-09-13",
                "2021-09-12",
                "2022-09-11",
            ],
            "home_team": [
                "XXX",
            ] * 3,
            "away_team": [
                "YYY",
            ] * 3,
            "home_coach": [
                "A Coach",
                "B Coach",
                "A Coach",
            ],
            "away_coach": [
                "Z Coach",
            ] * 3,
        }
    )

    stints = coaching.head_coach_stints(
        sched
    ).filter(
        pl.col("team") == "XXX"
    )

    assert stints.height == 3, (
        "a returning coach must be two stints, "
        "not one continuous tenure"
    )


def test_coach_id_expression_matches_the_python_helper():
    names = [
        "Mike O'Brien",
        "Bill Belichick",
        "Jim Schwartz-Smith",
        " Sean  McVay ",
    ]

    df = pl.DataFrame(
        {
            "coach_name": names,
        }
    )

    from_expr = (
        df.select(
            coaching.coach_id_expr()
        )
        .to_series()
        .to_list()
    )

    assert from_expr == [
        coaching.coach_id(name)
        for name in names
    ]


# --------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------

def _spec(name: str):
    return get_manifest()[name]


def test_primary_key_check_detects_duplicates():
    spec = _spec("schedules")

    df = pl.DataFrame(
        {
            "game_id": [
                "a",
                "a",
            ],
            "season": [
                2020,
                2020,
            ],
        }
    )

    result = quality.check_primary_key(
        df,
        spec,
    )

    assert not result.passed
    assert result.severity == "error"


def test_primary_key_check_detects_a_wrong_manifest_key():
    spec = _spec("schedules")

    df = pl.DataFrame(
        {
            "not_the_key": [
                1,
                2,
            ],
        }
    )

    result = quality.check_primary_key(
        df,
        spec,
    )

    assert not result.passed
    assert "not in the data" in result.detail


def test_contiguity_check_finds_a_missing_season():
    spec = _spec("injuries")

    df = pl.DataFrame(
        {
            "season": [
                2019,
                2020,
                2022,
            ],
            "week": [
                1,
                1,
                1,
            ],
            "team": [
                "KC",
            ] * 3,
            "gsis_id": [
                "x",
                "y",
                "z",
            ],
        }
    )

    result = quality.check_season_contiguity(
        df,
        spec,
    )

    assert not result.passed
    assert result.data["holes"] == [2021]


def test_contiguity_handles_float_seasons():
    """nflverse ships `season` as a float in several tables."""
    spec = _spec("injuries")

    df = pl.DataFrame(
        {
            "season": [
                2019.0,
                2020.0,
                2021.0,
            ],
        }
    )

    assert quality.check_season_contiguity(
        df,
        spec,
    ).passed


def test_null_rate_check_flags_a_column_that_did_not_exist_yet():
    spec = _spec("participation")

    df = pl.DataFrame(
        {
            "nflverse_game_id": [
                "g",
            ] * 100,
            "play_id": list(
                range(100)
            ),
            "defense_coverage_type": [
                None,
            ] * 100,
        }
    )

    result = quality.check_null_rates(
        df,
        spec,
    )

    assert result.severity == "warning"

    assert (
        "defense_coverage_type"
        in result.data["near_empty"]
    )


def test_injury_burden_excludes_post_kickoff_revision():
    """Post-kickoff injury revisions must never affect a pregame feature row."""
    from examples.pregame_frame import injury_burden

    kickoff = datetime(
        2023,
        9,
        10,
        17,
        0,
        tzinfo=UTC,
    )

    games = pl.DataFrame(
        {
            "game_id": [
                "g1",
            ],
            "season": [
                2023,
            ],
            "week": [
                1,
            ],
            "team": [
                "KC",
            ],
            "kickoff_utc": [
                kickoff,
            ],
        }
    )

    injuries = pl.DataFrame(
        {
            "season": [
                2023,
                2023,
                2023,
            ],
            "week": [
                1,
                1,
                1,
            ],
            "team": [
                "KC",
                "KC",
                "KC",
            ],
            "gsis_id": [
                "p1",
                "p1",
                "p2",
            ],
            "report_status": [
                "Questionable",
                "Out",
                "Out",
            ],
            "date_modified": [
                datetime(
                    2023,
                    9,
                    9,
                    18,
                    0,
                    tzinfo=UTC,
                ),
                datetime(
                    2023,
                    9,
                    10,
                    20,
                    0,
                    tzinfo=UTC,
                ),
                datetime(
                    2023,
                    9,
                    9,
                    19,
                    0,
                    tzinfo=UTC,
                ),
            ],
        }
    )

    out = injury_burden(
        injuries,
        games,
    )

    assert out.get_column(
        "n_out"
    ).to_list() == [1]

    assert out.get_column(
        "n_questionable"
    ).to_list() == [1]

    assert out.get_column(
        "n_doubtful"
    ).to_list() == [0]