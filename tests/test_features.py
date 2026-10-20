"""Hand-checked examples per feature family, on a tiny synthetic fixture."""

import math

import pandas as pd
import pytest

from nbaxp.features.build import (
    ELO_CARRY,
    ELO_MEAN,
    _State,
    build_features,
    haversine_km,
    load_arenas,
)

BOS, NYK, LAL, DEN = "1610612738", "1610612752", "1610612747", "1610612743"
CONF = {BOS: "East", NYK: "East", LAL: "West", DEN: "West"}


def game(gid, date, home, away, hs, as_, season="2020-21", st="Regular Season", series=None):
    return {
        "gameId": gid,
        "gameDate": pd.Timestamp(date),
        "season": season,
        "seasonType": st,
        "seriesId": series,
        "homeFranchiseId": home,
        "awayFranchiseId": away,
        "homeConference": CONF[home],
        "awayConference": CONF[away],
        "homeScore": hs,
        "awayScore": as_,
        "margin": hs - as_,
    }


def schedule_for(games_df):
    """Real schedule plus 13 dummy teams per conference so the 10th-team rule has 15 teams."""
    rows = [(s, t, CONF[t], 82) for s in games_df.season.unique() for t in CONF]
    rows += [(s, f"{c}{i}", c, 82) for s in games_df.season.unique() for c in ("East", "West") for i in range(13)]
    return pd.DataFrame(rows, columns=["season", "franchiseId", "conference", "n_scheduled"])


def minutes_for(games_df, players_by_team):
    rows = []
    for g in games_df.itertuples():
        for team in (g.homeFranchiseId, g.awayFranchiseId):
            for pid, m in players_by_team.get(team, {}).items():
                rows.append((g.gameId, g.gameDate, g.season, team, pid, m))
    return pd.DataFrame(rows, columns=["gameId", "gameDate", "season", "franchiseId", "playerId", "minutes"])


def run(games, players_by_team=None):
    df = pd.DataFrame(games)
    return build_features(df, minutes_for(df, players_by_team or {}), schedule_for(df)).set_index("gameId")


# BOS: home win, then away win (back-to-back), then home win, then a home game 4 days later.
BASE = [
    game("g1", "2021-01-01", BOS, NYK, 110, 100),
    game("g2", "2021-01-02", LAL, BOS, 100, 105),
    game("g3", "2021-01-02", NYK, DEN, 90, 95),
    game("g4", "2021-01-05", BOS, LAL, 100, 98),
    game("g5", "2021-01-09", BOS, NYK, 100, 99),
]
PLAYERS = {BOS: {**{f"p{i}": 30.0 for i in range(1, 9)}, "p9": 5.0}}


def test_elo_hand_computed():
    f = run(BASE[:1] + [game("g9", "2021-10-20", BOS, NYK, 100, 99, season="2021-22")])
    # g1: both 1505, home advantage 100 -> expected 0.6401, margin multiplier 13^0.8/(7.5+0.6)=0.9609, delta 6.917
    assert f.loc["g1", ["home_elo", "away_elo"]].tolist() == [1505.0, 1505.0]
    # next season: ratings are pulled 25% toward the mean before the game
    assert f.loc["g9", "home_elo"] == pytest.approx(ELO_CARRY * 1511.9171 + (1 - ELO_CARRY) * ELO_MEAN, abs=1e-3)
    assert f.loc["g9", "away_elo"] == pytest.approx(ELO_CARRY * 1498.0829 + (1 - ELO_CARRY) * ELO_MEAN, abs=1e-3)
    assert f.loc["g9", "diff_elo"] == pytest.approx(ELO_CARRY * 2 * 6.9171, abs=1e-3)


def test_record_streak_point_differential():
    f = run(BASE)
    assert f.loc["g1", "home_win_pct"] != f.loc["g1", "home_win_pct"]  # NaN: no games yet
    assert f.loc["g4", ["home_wins", "home_losses", "home_win_pct"]].tolist() == [2, 0, 1.0]
    assert f.loc["g4", ["away_wins", "away_losses"]].tolist() == [0, 1]  # LAL lost g2
    assert f.loc["g4", "home_streak"] == 2 and f.loc["g4", "away_streak"] == -1
    # BOS margins before g4: +10 (g1) and +5 (g2, won away by 5)
    assert f.loc["g4", "home_pd_last5"] == 7.5 and f.loc["g4", "home_pd_last10"] == 7.5
    assert f.loc["g4", "away_pd_last5"] == -5.0


def test_streak_flips_sign_on_a_loss():
    f = run(BASE + [game("g6", "2021-01-12", NYK, BOS, 100, 90), game("g7", "2021-01-14", BOS, DEN, 90, 95)])
    assert f.loc["g6", "away_streak"] == 4  # BOS won g1, g2, g4, g5
    assert f.loc["g7", "home_streak"] == -1


def test_rest_and_back_to_back():
    f = run(BASE)
    assert f.loc["g2", "away_rest_days"] == 1 and f.loc["g2", "away_back_to_back"]  # BOS played on 01-01
    assert f.loc["g4", "home_rest_days"] == 3 and not f.loc["g4", "home_back_to_back"]
    assert math.isnan(f.loc["g1", "home_rest_days"]) and not f.loc["g1", "home_back_to_back"]


def test_travel_includes_trip_to_this_game_and_window_cutoff():
    f = run(BASE)
    bos_la = haversine_km(load_arenas()[BOS], load_arenas()[LAL])
    assert 4100 < bos_la < 4200  # Boston-Los Angeles great circle, about 4170 km
    assert f.loc["g1", "home_travel_km_3d"] == 0  # season opener at home
    assert 280 < f.loc["g1", "away_travel_km_3d"] < 330  # NYK to Boston, about 300 km
    assert f.loc["g2", "away_travel_km_3d"] == pytest.approx(bos_la)
    # g4 on 01-05: windows start 01-02, so the g2 trip (01-02) counts and the g4 trip home counts
    assert f.loc["g4", "home_travel_km_3d"] == pytest.approx(2 * bos_la)
    # g5 on 01-09: the 3-day window starts 01-06 (only this home game, 0 km); the 7-day window starts 01-02
    assert f.loc["g5", "home_travel_km_3d"] == 0
    assert f.loc["g5", "home_travel_km_7d"] == pytest.approx(2 * bos_la)


def test_rotation_minutes_load():
    f = run(BASE, PLAYERS)
    # p9 (5 min) is outside the top 8. Before g4 (01-05): g2 on 01-02 is inside 3 days, g1 on 01-01 is not.
    assert f.loc["g4", "home_rotation_min_3d"] == 8 * 30
    assert f.loc["g4", "home_rotation_min_7d"] == 16 * 30
    assert f.loc["g4", "home_rotation_min_14d"] == 16 * 30
    assert math.isnan(f.loc["g1", "home_rotation_min_3d"])
    # same-day rule: g2 and g3 are on 01-02, BOS g1 minutes only (30*8)
    assert f.loc["g2", "away_rotation_min_3d"] == 8 * 30


def test_eliminated_rule():
    teams = [(f"E{i}", "East") for i in range(15)]
    sched = pd.DataFrame(
        [("2020-21", t, c, 82) for t, c in teams], columns=["season", "franchiseId", "conference", "n_scheduled"]
    )
    s = _State(load_arenas(), sched)
    for i in range(10):
        s.wins[("2020-21", f"E{i}")], s.losses[("2020-21", f"E{i}")] = 40, 20  # 10th team has 40 wins
    s.wins[("2020-21", "E11")], s.losses[("2020-21", "E11")] = 0, 42  # max 0 + 40 = 40, not below 40
    s.wins[("2020-21", "E12")], s.losses[("2020-21", "E12")] = 0, 43  # max 39 < 40
    s.wins[("2020-21", "E13")], s.losses[("2020-21", "E13")] = 5, 50  # max 5 + 27 = 32 < 40
    s.wins[("2020-21", "E14")], s.losses[("2020-21", "E14")] = 30, 30  # max 30 + 22 = 52
    elim = {t: s._eliminated(t, "East", "2020-21") for t, _ in teams}
    assert [t for t, v in elim.items() if v] == ["E12", "E13"]
    s.sched[("2020-21", "E14")] = 70  # shortened schedule: max 30 + 10 = 40, still not below
    assert not s._eliminated("E14", "East", "2020-21")
    s.sched[("2020-21", "E14")] = 69  # max 39 < 40
    assert s._eliminated("E14", "East", "2020-21")


def test_win_or_go_home_playoffs():
    # The home team wins every game. Home teams: BOS, BOS, NYK, NYK, BOS, NYK, BOS.
    homes = [BOS, BOS, NYK, NYK, BOS, NYK, BOS]
    games = [
        game(f"p{i}", f"2021-06-{i:02d}", h, NYK if h == BOS else BOS, 100, 90, st="Playoffs", series="S")
        for i, h in enumerate(homes, 1)
    ]
    f = run(games)
    # Before game 6: BOS lost p3 and p4 (2 losses); NYK lost p1, p2 and p5 (3 losses). NYK is home in p6.
    assert f.loc["p6", ["home_win_or_go_home", "away_win_or_go_home"]].tolist() == [True, False]
    assert f.loc["p5", ["home_win_or_go_home", "away_win_or_go_home"]].tolist() == [False, False]
    # game 7: 3-3, both teams have 3 losses
    assert f.loc["p7", ["home_win_or_go_home", "away_win_or_go_home"]].tolist() == [True, True]
