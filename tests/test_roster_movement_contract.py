from datetime import UTC, date, datetime

import polars as pl

from nfldl import pit
from nfldl.manifest import load_manifest


def test_season_rosters_are_not_direct_point_in_time_pregame_data():
    """Season roster files are snapshots, not historical transaction ledgers."""
    assert pit.DATASET_AVAILABILITY["rosters"] == "unknown"


def test_historical_weekly_rosters_do_not_claim_same_week_pregame_knowledge():
    """Week identity does not prove when the historical snapshot was published."""
    spec = load_manifest()["weekly_rosters"]

    assert pit.DATASET_AVAILABILITY["weekly_rosters"] == "unknown"
    assert spec.timestamp_field is None
    assert spec.raw.get("point_in_time") == "week_snapshot"


def test_trades_declare_the_real_date_field_and_date_granularity():
    spec = load_manifest()["trades"]

    assert spec.timestamp_field == "trade_date"
    assert spec.raw.get("point_in_time") == "date"


def test_date_only_trade_is_not_treated_as_known_at_start_of_trade_day():
    """A date-only transaction cannot safely leak into an earlier same-day game."""
    trade = pl.DataFrame(
        {
            "trade_date": [date(2023, 10, 10)],
            "kickoff_utc": [
                datetime(
                    2023,
                    10,
                    10,
                    20,
                    0,
                    tzinfo=UTC,
                )
            ],
        }
    )

    out = pit.knowledge_time(
        trade,
        "trades",
        report_col="trade_date",
    )

    assert (
        out.get_column("_known_at")[0]
        > out.get_column("kickoff_utc")[0]
    )


def test_contracts_remain_unknown_for_historical_point_in_time_use():
    spec = load_manifest()["contracts"]

    assert pit.DATASET_AVAILABILITY["contracts"] == "unknown"
    assert spec.timestamp_field is None
    assert spec.raw.get("leakage_risk") == "high"


def test_trade_primary_key_distinguishes_named_players_without_pfr_ids():
    spec = load_manifest()["trades"]

    assert spec.primary_key == (
        "trade_id",
        "season",
        "gave",
        "received",
        "pick_season",
        "pick_round",
        "pick_number",
        "pfr_id",
        "pfr_name",
    )


def test_fetch_trades_removes_only_exact_upstream_duplicates(monkeypatch):
    from nfldl.sources import nflverse

    raw = pl.DataFrame(
        {
            "trade_id": [62, 62, 62],
            "season": [2004, 2004, 2004],
            "trade_date": [
                date(2004, 10, 19),
                date(2004, 10, 19),
                date(2004, 10, 19),
            ],
            "gave": ["SD", "SD", "SD"],
            "received": ["TB", "TB", "TB"],
            "pick_season": [None, None, None],
            "pick_round": [None, None, None],
            "pick_number": [None, None, None],
            "conditional": [None, None, None],
            "pfr_id": ["", "", ""],
            "pfr_name": [
                "Chris Colmer",
                "Chris Colmer",
                "Andrew Hoffman",
            ],
        }
    )

    monkeypatch.setattr(
        nflverse.nfl,
        "load_trades",
        lambda: raw,
    )

    out = nflverse.fetch_trades()

    assert out.height == 2
    assert sorted(out.get_column("pfr_name").to_list()) == [
        "Andrew Hoffman",
        "Chris Colmer",
    ]


def test_trade_nullable_key_contract_is_conditional():
    spec = load_manifest()["trades"]

    rules = {
        (rule.column, tuple(rule.when))
        for rule in spec.nullable_key_rules
    }

    assert rules == {
        ("pick_season", (("conditional", None),)),
        ("pick_round", (("pick_number", None),)),
        ("pick_number", (("conditional", None),)),
        ("pick_number", (("pfr_name", ""),)),
    }


def test_global_nullable_key_rules_allow_valid_trade_asset_nulls():
    from nfldl import quality

    spec = load_manifest()["trades"]

    df = pl.DataFrame(
        {
            "trade_id": [1, 2, 3],
            "season": [2025, 2025, 2025],
            "trade_date": [
                date(2025, 3, 1),
                date(2025, 3, 2),
                date(2025, 3, 3),
            ],
            "gave": ["CHI", "GB", "DET"],
            "received": ["BUF", "NYJ", "SEA"],
            "pick_season": [None, 2026, 2026],
            "pick_round": [None, None, 4],
            "pick_number": [None, None, None],
            "conditional": [None, 1, 0],
            "pfr_id": ["Play01", "", ""],
            "pfr_name": ["Example Player", "", ""],
        }
    )

    result = quality.check_null_rates(df, spec)

    assert result.passed
    assert result.severity != "error"


def test_global_nullable_key_rules_reject_invalid_trade_pick_season_null():
    from nfldl import quality

    spec = load_manifest()["trades"]

    df = pl.DataFrame(
        {
            "trade_id": [1],
            "season": [2025],
            "trade_date": [date(2025, 3, 1)],
            "gave": ["CHI"],
            "received": ["BUF"],
            "pick_season": [None],
            "pick_round": [4],
            "pick_number": [120],
            "conditional": [0],
            "pfr_id": [""],
            "pfr_name": [""],
        }
    )

    result = quality.check_null_rates(df, spec)

    assert not result.passed
    assert result.severity == "error"
    assert "pick_season" in result.data["key_nulls"]



def test_contracts_do_not_claim_a_false_logical_primary_key():
    """The OTC current-state table has no verified stable contract-row identity."""
    spec = load_manifest()["contracts"]

    assert spec.required is False
    assert spec.primary_key == ()


def test_fetch_contracts_removes_only_exact_upstream_duplicates(monkeypatch):
    """Distinct current-state rows must survive even when old key fields collide."""
    from nfldl.sources import nflverse

    raw = pl.DataFrame(
        {
            "player": [
                "Example Player",
                "Example Player",
                "Example Player",
            ],
            "position": [
                "WR",
                "WR",
                "PR",
            ],
            "team": [
                "Example",
                "Example",
                "Example",
            ],
            "is_active": [
                False,
                False,
                False,
            ],
            "year_signed": [
                2022,
                2022,
                2022,
            ],
            "years": [
                1,
                1,
                1,
            ],
            "value": [
                1.0,
                1.0,
                1.0,
            ],
            "apy": [
                1.0,
                1.0,
                1.0,
            ],
            "guaranteed": [
                0.0,
                0.0,
                0.0,
            ],
            "otc_id": [
                123,
                123,
                123,
            ],
        }
    )

    monkeypatch.setattr(
        nflverse.nfl,
        "load_contracts",
        lambda: raw,
    )

    out = nflverse.fetch_contracts()

    assert out.height == 2
    assert sorted(out.get_column("position").to_list()) == [
        "PR",
        "WR",
    ]



def test_contracts_explicitly_declare_current_snapshot_semantics():
    """Retrospective contract fields must not masquerade as historical PIT data."""
    spec = load_manifest()["contracts"]

    assert spec.raw.get("point_in_time") == "current_snapshot"

    caveat = str(spec.raw.get("coverage_caveat", "")).lower()

    assert "historical pregame" in caveat
    assert "year_signed" in caveat
    assert "does not establish" in caveat
