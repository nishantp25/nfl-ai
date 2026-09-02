"""Worked example: assemble a pregame frame that provably cannot see the future.

This is the payoff for everything else in the package. It shows the intended
shape of downstream feature work:

    canonical raw  ->  derive features ourselves  ->  as-of join  ->  audit

Run after ingesting at least `schedules`, `play_by_play` and `injuries`:

    nfldl ingest schedules play_by_play injuries -r 2021 2023
    python examples/pregame_frame.py

Two things it demonstrates deliberately:

* **Team strength is computed here, not downloaded.** The manifest forbids
  fetching Elo or opponent-adjusted EPA from a third party precisely so that
  every rating is a function of games that had already been played at the time.
* **The audit is the last step and it can fail.** A frame that passes
  `pit.audit_frame` has no outcome columns, no unlagged postgame measures and no
  ingest bookkeeping masquerading as a feature.
"""

from __future__ import annotations

import logging

import polars as pl

from nfldl import pit
from nfldl.manifest import get_manifest
from nfldl.storage import read_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("example")

SEASONS = [2021, 2022, 2023]


def team_epa_by_game(pbp: pl.DataFrame) -> pl.DataFrame:
    """Offensive EPA per play and success rate, per team per game.

    Derived from canonical play-by-play, which is the whole point: the same code
    reconstructs these numbers for any historical week without a third-party
    file that may have been recomputed since.
    """
    plays = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("epa").is_not_null()
        & (pl.col("play_type").is_in(["pass", "run"]))
    )

    return (
        plays.group_by(["game_id", "season", "week", "posteam"])
        .agg(
            pl.col("epa").mean().alias("off_epa_per_play"),
            (pl.col("epa") > 0).mean().alias("off_success_rate"),
            pl.len().alias("off_plays"),
            (pl.col("pass") == 1).mean().alias("pass_rate"),
        )
        .rename({"posteam": "team"})
    )


def rolling_form(per_game: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Trailing form that excludes the current game.

    ``shift(1)`` before the rolling mean is the entire safety property: at week
    W a team's form reflects weeks < W only. Without the shift this feature
    would contain the result of the game being predicted.
    """
    ordered = per_game.sort(["team", "season", "week"])

    return ordered.with_columns(
        [
            pl.col(c)
            .shift(1)
            .rolling_mean(window_size=window, min_samples=1)
            .over(["team", "season"])
            .alias(f"{c}_prior{window}")
            for c in ("off_epa_per_play", "off_success_rate", "pass_rate")
        ]
    )


def injury_burden(
    injuries: pl.DataFrame,
    game_cutoffs: pl.DataFrame,
) -> pl.DataFrame:
    """Count injury designations that were actually known before kickoff.

    nflverse injury rows may be revised after a game has already started or
    finished. Grouping the entire season/week table before applying a cutoff
    could therefore leak later information into a historical pregame feature.

    For every game/team combination this function:

    1. attaches the game's actual kickoff timestamp;
    2. removes injury revisions with ``date_modified >= kickoff_utc``;
    3. keeps the latest surviving status for each player;
    4. only then aggregates Out, Doubtful and Questionable counts.

    This makes the injury feature point-in-time correct rather than merely
    week-aligned.
    """
    required = {
        "season",
        "week",
        "team",
        "gsis_id",
        "report_status",
        "date_modified",
    }

    if not required.issubset(injuries.columns):
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "season": pl.Int64,
                "week": pl.Int64,
                "team": pl.String,
                "n_out": pl.UInt32,
                "n_doubtful": pl.UInt32,
                "n_questionable": pl.UInt32,
            }
        )

    cutoffs = (
        game_cutoffs.select(
            "game_id",
            "season",
            "week",
            "team",
            "kickoff_utc",
        )
        .with_columns(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
        )
    )

    known = (
        injuries.with_columns(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("date_modified")
            .cast(pl.Datetime("us", "UTC"))
            .alias("_known_at"),
        )
        .join(
            cutoffs,
            on=["season", "week", "team"],
            how="inner",
        )
        .filter(
            pl.col("_known_at").is_not_null()
            & (pl.col("_known_at") < pl.col("kickoff_utc"))
        )
        .sort(
            [
                "game_id",
                "team",
                "gsis_id",
                "_known_at",
            ]
        )
        .unique(
            subset=["game_id", "team", "gsis_id"],
            keep="last",
            maintain_order=True,
        )
    )

    # Defense in depth: if the filtering logic above ever changes incorrectly,
    # this assertion should immediately expose the leakage.
    pit.assert_no_future_rows(known)

    return (
        known.group_by(
            [
                "game_id",
                "season",
                "week",
                "team",
            ]
        )
        .agg(
            (pl.col("report_status") == "Out")
            .sum()
            .alias("n_out"),
            (pl.col("report_status") == "Doubtful")
            .sum()
            .alias("n_doubtful"),
            (pl.col("report_status") == "Questionable")
            .sum()
            .alias("n_questionable"),
        )
    )


def main() -> None:
    mf = get_manifest()

    games = read_dataset(
        mf["schedules"],
        seasons=SEASONS,
    )

    pbp = read_dataset(
        mf["play_by_play"],
        seasons=SEASONS,
    )

    injuries = read_dataset(
        mf["injuries"],
        seasons=SEASONS,
    )

    per_game = rolling_form(
        team_epa_by_game(pbp)
    )

    # Attach kickoff to the postgame measures so knowledge time is a timestamp,
    # not a week number.
    per_game = per_game.join(
        games.select(
            "game_id",
            "kickoff_utc",
        ),
        on="game_id",
        how="inner",
    )

    per_game = pit.knowledge_time(
        per_game,
        "play_by_play",
    )

    form_cols = [
        c
        for c in per_game.columns
        if c.endswith("prior5")
    ]

    history = per_game.select(
        [
            "team",
            "_known_at",
            *form_cols,
        ]
    )

    # One row per team per game -- home and away handled symmetrically.
    long_games = pl.concat(
        [
            games.select(
                "game_id",
                "season",
                "week",
                "kickoff_utc",
                pl.col("home_team").alias("team"),
                pl.lit(True).alias("is_home"),
            ),
            games.select(
                "game_id",
                "season",
                "week",
                "kickoff_utc",
                pl.col("away_team").alias("team"),
                pl.lit(False).alias("is_home"),
            ),
        ]
    )

    # Team form is joined strictly backward in time.
    frame = pit.asof_join(
        long_games,
        history,
        by=["team"],
    )

    pit.assert_no_future_rows(frame)

    # Injury information must also obey the actual kickoff cutoff.
    injuries_pit = injury_burden(
        injuries,
        long_games,
    )

    frame = frame.join(
        injuries_pit,
        on=[
            "game_id",
            "season",
            "week",
            "team",
        ],
        how="left",
    )

    feature_cols = [
        c
        for c in frame.columns
        if c.endswith("prior5")
    ] + [
        "n_out",
        "n_doubtful",
        "n_questionable",
        "is_home",
    ]

    features = frame.select(
        "game_id",
        "season",
        "week",
        "team",
        *[
            c
            for c in feature_cols
            if c in frame.columns
        ],
    )

    origin = {
        c: "play_by_play"
        for c in features.columns
        if c.endswith("prior5")
    }

    origin.update(
        {
            c: "injuries"
            for c in (
                "n_out",
                "n_doubtful",
                "n_questionable",
            )
            if c in features.columns
        }
    )

    pit.audit_frame(
        features,
        dataset_of=origin,
    )

    log.info(
        "pregame frame: %d rows x %d columns",
        features.height,
        features.width,
    )

    print(
        features.head(8)
    )

    # And the demonstration that the guard actually bites.
    leaky = features.join(
        games.select(
            "game_id",
            "result",
        ),
        on="game_id",
        how="left",
    )

    try:
        pit.audit_frame(
            leaky,
            dataset_of=origin,
        )
    except pit.LeakageError as exc:
        log.info(
            "audit correctly rejected the leaky variant:\n%s",
            exc,
        )


if __name__ == "__main__":
    main()