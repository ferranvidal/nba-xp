"""As-of rule on real staged data: removing all games on or after D changes no feature of the games on D."""

from pathlib import Path

import pandas as pd
import pytest

from nbaxp.features.build import build_features, season_schedule

STAGED = Path(__file__).resolve().parents[1] / "data" / "staged"
pytestmark = pytest.mark.skipif(not (STAGED / "games.parquet").exists(), reason="run `nbaxp stage` first")

SEASONS = ["2022-23", "2023-24"]
# regular season, play-in (two games that day), playoffs, NBA Cup final
DATES = ["2023-01-15", "2023-04-12", "2024-05-09", "2023-12-09"]


@pytest.fixture(scope="module")
def data():
    games = pd.read_parquet(STAGED / "games.parquet")
    minutes = pd.read_parquet(STAGED / "player_minutes.parquet")
    games = games[games.season.isin(SEASONS)]
    minutes = minutes[minutes.gameId.isin(games.gameId)]
    schedule = season_schedule(games)  # the schedule is known in advance, so it does not shrink with D
    return games, minutes, schedule, build_features(games, minutes, schedule)


@pytest.mark.parametrize("date", DATES)
def test_features_on_d_ignore_games_on_or_after_d(data, date):
    games, minutes, schedule, full = data
    d = pd.Timestamp(date)
    day = pd.to_datetime(games.gameDate)
    # Keep D's games but drop every later game and the minutes of D's games, and flip the
    # results of D's games. Features of D's games must not change.
    cut = games[day <= d].copy()
    on_d_mask = pd.to_datetime(cut.gameDate) == d
    cut.loc[on_d_mask, ["homeScore", "awayScore"]] = cut.loc[on_d_mask, ["awayScore", "homeScore"]].to_numpy()
    cut["margin"] = cut.homeScore - cut.awayScore
    cut_minutes = minutes[minutes.gameId.isin(cut.gameId[~on_d_mask])]
    cut = build_features(cut, cut_minutes, schedule)
    on_d = full[full.gameDate == d].drop(columns="margin").reset_index(drop=True)
    cut_d = cut[cut.gameDate == d].drop(columns="margin").reset_index(drop=True)
    assert len(on_d) > 0
    assert on_d[["home_player_strength", "away_availability"]].notna().any().all()  # the new columns are compared
    pd.testing.assert_frame_equal(on_d, cut_d)
