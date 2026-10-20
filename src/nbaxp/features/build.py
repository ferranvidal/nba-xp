"""Staged data -> one feature row per game.

AS-OF RULE: the features of a game on date D use only games with gameDate < D.
The loop below handles one date at a time: it computes the features of every game
on D from the current state, and only then adds the results of D to the state.
"""

import math
from collections import defaultdict
from importlib import resources
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

# Elo (538-style NBA Elo)
ELO_INIT = 1505.0
ELO_MEAN = 1505.0
ELO_K = 20.0
ELO_HOME_ADV = 100.0  # Elo points; 0 at neutral sites (NBA Cup final, bubble)
ELO_CARRY = 0.75  # share of the rating kept at the start of a new season

LAS_VEGAS = (36.1029, -115.1784)  # T-Mobile Arena, NBA Cup final
BUBBLE_START = pd.Timestamp("2020-07-01")  # 2019-20 restart in Orlando

ROTATION_GAMES = 10
ROTATION_SIZE = 8
LOAD_DAYS = (14, 7, 3)  # longest window first
TRAVEL_DAYS = (3, 7)

# Feature name -> has a home-minus-away difference column
FEATURES = {
    "elo": True,
    "wins": True,
    "losses": True,
    "win_pct": True,
    "streak": True,
    "pd_last5": True,
    "pd_last10": True,
    "rest_days": True,
    "back_to_back": False,
    "travel_km_3d": True,
    "travel_km_7d": True,
    "rotation_min_3d": True,
    "rotation_min_7d": True,
    "rotation_min_14d": True,
    "eliminated": False,
    "win_or_go_home": False,
}


def load_arenas() -> dict[str, tuple[float, float]]:
    with resources.files("nbaxp.data").joinpath("arenas.csv").open() as f:
        df = pd.read_csv(f, dtype={"franchiseId": str})
    return {r.franchiseId: (r.lat, r.lon) for r in df.itertuples()}


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def season_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games per team and season, and the team's conference.

    The schedule is known before the season starts, so this is not outcome leakage.
    """
    rs = games[games.seasonType == "Regular Season"]
    sides = [
        rs.rename(columns={"homeFranchiseId": "franchiseId", "homeConference": "conference"}),
        rs.rename(columns={"awayFranchiseId": "franchiseId", "awayConference": "conference"}),
    ]
    both = pd.concat([s[["season", "franchiseId", "conference"]] for s in sides])
    return both.groupby(["season", "franchiseId", "conference"]).size().rename("n_scheduled").reset_index()


def _elo_pre(elo: dict, elo_season: dict, team: str, season: str) -> float:
    if team not in elo:
        return ELO_INIT
    if elo_season[team] != season:
        return ELO_CARRY * elo[team] + (1 - ELO_CARRY) * ELO_MEAN
    return elo[team]


class _State:
    """Everything known after processing all games before the current date."""

    def __init__(self, arenas, schedule: pd.DataFrame):
        self.arenas = arenas
        self.sched = {(r.season, r.franchiseId): r.n_scheduled for r in schedule.itertuples()}
        self.conf_teams = defaultdict(list)  # (season, conference) -> franchiseIds
        for r in schedule.itertuples():
            self.conf_teams[(r.season, r.conference)].append(r.franchiseId)
        self.elo, self.elo_season = {}, {}
        self.log = defaultdict(list)  # team -> list of finished games (dict), oldest first
        self.player_log = defaultdict(list)  # player -> [(day, minutes)]
        self.wins = defaultdict(int)  # (season, team), regular season only
        self.losses = defaultdict(int)
        self.streak = {}  # team -> (season, signed streak)
        self.series_losses = defaultdict(int)  # (seriesId, team)
        self.playin_games = defaultdict(int)  # (season, team)

    # ---- features -------------------------------------------------------------------
    def team_features(self, team: str, g: dict, loc: tuple[float, float]) -> dict:
        season, day = g["season"], g["day"]
        hist = self.log[team]
        f = {"elo": _elo_pre(self.elo, self.elo_season, team, season)}

        w, l_ = self.wins[(season, team)], self.losses[(season, team)]
        f.update(wins=w, losses=l_, win_pct=w / (w + l_) if w + l_ else np.nan)
        s_season, s_val = self.streak.get(team, (None, 0))
        f["streak"] = s_val if s_season == season else 0

        recent = []
        for h in reversed(hist):
            if h["season"] != season or len(recent) == 10:
                break
            recent.append(h["margin"])
        f["pd_last5"] = sum(recent[:5]) / len(recent[:5]) if recent else np.nan
        f["pd_last10"] = sum(recent) / len(recent) if recent else np.nan

        in_season = bool(hist) and hist[-1]["season"] == season
        rest = day - hist[-1]["day"] if in_season else np.nan
        f["rest_days"] = rest
        f["back_to_back"] = bool(rest == 1)

        # A team starts a season at home. The trip to this game counts in every window.
        prev_loc = hist[-1]["loc"] if in_season else self.arenas[team]
        leg = haversine_km(prev_loc, loc)
        for n in TRAVEL_DAYS:
            total = leg
            for h in reversed(hist):
                if h["day"] < day - n:
                    break
                total += h["leg"]
            f[f"travel_km_{n}d"] = total

        last = hist[-ROTATION_GAMES:]
        if last:
            tot = defaultdict(float)
            for h in last:
                for pid, m in h["players"]:
                    tot[pid] += m
            rotation = sorted(tot, key=lambda p: (-tot[p], p))[:ROTATION_SIZE]
            for n in LOAD_DAYS:
                f[f"rotation_min_{n}d"] = 0.0
            for pid in rotation:
                for d, m in reversed(self.player_log[pid]):
                    if d < day - LOAD_DAYS[0]:
                        break
                    for n in LOAD_DAYS:
                        if d >= day - n:
                            f[f"rotation_min_{n}d"] += m
        else:
            for n in LOAD_DAYS:
                f[f"rotation_min_{n}d"] = np.nan
        return f

    def _eliminated(self, team: str, conference: str, season: str) -> bool:
        """Max possible wins < current wins of the conference's last postseason spot (ignores tiebreakers).

        The last spot is 10th since the play-in started in 2020-21, and 8th before.
        """
        teams = self.conf_teams[(season, conference)]
        last_spot = 10 if season >= "2020-21" else 8
        cutoff = sorted((self.wins[(season, t)] for t in teams), reverse=True)[last_spot - 1]
        played = self.wins[(season, team)] + self.losses[(season, team)]
        return self.wins[(season, team)] + self.sched[(season, team)] - played < cutoff

    def _win_pct_rank(self, conference: str, season: str) -> dict[str, int]:
        teams = self.conf_teams[(season, conference)]

        def key(t):
            w, l_ = self.wins[(season, t)], self.losses[(season, t)]
            return (-(w / (w + l_) if w + l_ else 0.0), t)

        return {t: i + 1 for i, t in enumerate(sorted(teams, key=key))}

    def flags(self, game: dict) -> tuple[dict, dict]:
        """eliminated and win_or_go_home for the home and the away team."""
        season, st = game["season"], game["seasonType"]
        out = []
        for side in ("home", "away"):
            team = game[f"{side}FranchiseId"]
            conf = game[f"{side}Conference"]
            elim = st == "Regular Season" and self._eliminated(team, conf, season)
            wgh = False
            if st == "Playoffs":
                wgh = self.series_losses[(game["seriesId"], team)] == 3
            elif st == "Play-In":
                if self.playin_games[(season, team)] > 0:
                    wgh = True  # final play-in game of the conference
                else:
                    # First-round play-in game. The 7-v-8 and 9-v-10 pairings are public
                    # before tip-off (final regular-season standings), so using them is not
                    # outcome leakage. Rank is by regular-season win % before D.
                    rank = self._win_pct_rank(conf, season)
                    pair = (rank[game["homeFranchiseId"]] + rank[game["awayFranchiseId"]]) / 2
                    wgh = pair > 8.5  # 9-v-10 game
            out.append({"eliminated": bool(elim), "win_or_go_home": bool(wgh)})
        return out[0], out[1]

    # ---- update after a date is finished -------------------------------------------------
    def update(self, g: dict, loc, neutral: bool, players: dict) -> None:
        season, day = g["season"], g["day"]
        h, a = g["homeFranchiseId"], g["awayFranchiseId"]
        margin = g["margin"]

        eh, ea = (_elo_pre(self.elo, self.elo_season, t, season) for t in (h, a))
        dr = eh + (0.0 if neutral else ELO_HOME_ADV) - ea
        expected = 1 / (1 + 10 ** (-dr / 400))
        dr_winner = dr if margin > 0 else -dr
        mult = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * dr_winner)
        delta = ELO_K * mult * ((1.0 if margin > 0 else 0.0) - expected)
        self.elo[h], self.elo[a] = eh + delta, ea - delta
        self.elo_season[h] = self.elo_season[a] = season

        for team, sign, opp in ((h, 1, a), (a, -1, h)):
            won = margin * sign > 0
            prev_loc = self.log[team][-1]["loc"] if self.log[team] and self.log[team][-1]["season"] == season else None
            leg = haversine_km(prev_loc or self.arenas[team], loc)
            self.log[team].append(
                {
                    "day": day,
                    "season": season,
                    "margin": margin * sign,
                    "loc": loc,
                    "leg": leg,
                    "players": players[team],
                }
            )
            s_season, s_val = self.streak.get(team, (None, 0))
            s_val = s_val if s_season == season else 0
            self.streak[team] = (season, (max(s_val, 0) + 1) if won else (min(s_val, 0) - 1))
            if g["seasonType"] == "Regular Season":
                (self.wins if won else self.losses)[(season, team)] += 1
            elif g["seasonType"] == "Playoffs" and not won:
                self.series_losses[(g["seriesId"], team)] += 1
            elif g["seasonType"] == "Play-In":
                self.playin_games[(season, team)] += 1
            for pid, m in players[team]:
                self.player_log[pid].append((day, m))


def build_features(games: pd.DataFrame, minutes: pd.DataFrame, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per game. `schedule` defaults to the one computed from `games`."""
    if schedule is None:
        schedule = season_schedule(games)
    arenas = load_arenas()
    state = _State(arenas, schedule)

    games = games.assign(gameDate=pd.to_datetime(games.gameDate)).sort_values(["gameDate", "gameId"])
    games = games.reset_index(drop=True)
    games["day"] = (games.gameDate - pd.Timestamp("1970-01-01")).dt.days
    mins = defaultdict(list)
    for gid, fid, pid, m in zip(*(minutes[c].tolist() for c in ("gameId", "franchiseId", "playerId", "minutes"))):
        mins[(gid, fid)].append((pid, m))

    rows = []
    for _, day_games in groupby(games.to_dict("records"), key=lambda r: r["day"]):
        pending = []
        for g in day_games:
            is_cup = g["seasonType"] == "NBA Cup"
            is_bubble = g["season"] == "2019-20" and g["gameDate"] >= BUBBLE_START
            loc = LAS_VEGAS if is_cup else arenas[g["homeFranchiseId"]]
            row = {
                "gameId": g["gameId"],
                "gameDate": g["gameDate"],
                "season": g["season"],
                "seasonType": g["seasonType"],
                "homeFranchiseId": g["homeFranchiseId"],
                "awayFranchiseId": g["awayFranchiseId"],
                "margin": g["margin"],
                "is_regular_season": g["seasonType"] == "Regular Season",
                "is_play_in": g["seasonType"] == "Play-In",
                "is_playoffs": g["seasonType"] == "Playoffs",
                "is_nba_cup": is_cup,
                "is_bubble": is_bubble,
            }
            fl = dict(zip(("home", "away"), state.flags(g)))
            for side in ("home", "away"):
                f = state.team_features(g[f"{side}FranchiseId"], g, loc)
                f.update(fl[side])
                row.update({f"{side}_{k}": v for k, v in f.items()})
            for k, has_diff in FEATURES.items():
                if has_diff:
                    row[f"diff_{k}"] = row[f"home_{k}"] - row[f"away_{k}"]
            rows.append(row)
            pending.append((g, loc, is_cup or is_bubble))
        for g, loc, neutral in pending:
            players = {t: mins[(g["gameId"], t)] for t in (g["homeFranchiseId"], g["awayFranchiseId"])}
            state.update(g, loc, neutral, players)
    return pd.DataFrame(rows)


def build(data_dir: Path = Path("data")) -> int:
    games = pd.read_parquet(data_dir / "staged" / "games.parquet")
    minutes = pd.read_parquet(data_dir / "staged" / "player_minutes.parquet")
    feats = build_features(games, minutes)
    out = data_dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out / "features.parquet", index=False)
    return len(feats)
