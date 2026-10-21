"""Raw snapshot -> staged Parquet (games, player minutes). No data-quality checks."""

from pathlib import Path

import duckdb

FIRST_SEASON_YEAR = 2013  # season 2013-14


def latest_snapshot(data_dir: Path) -> Path:
    snaps = sorted((data_dir / "raw").glob("snapshot=*"))
    if not snaps:
        raise FileNotFoundError(f"no snapshot under {data_dir / 'raw'}")
    return snaps[-1]


def stage(data_dir: Path = Path("data"), snapshot: Path | None = None) -> dict[str, int]:
    snap = snapshot or latest_snapshot(data_dir)
    out = data_dir / "staged"
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for t in ("games", "teams", "player_stats"):
        con.execute(f"create view {t} as select * from read_parquet('{snap}/{t}/*.parquet')")
    # Conference is stored on the current team row only; take it per franchise.
    con.execute("create view conf as select franchiseId, max(conference) as conference from teams group by 1")
    con.execute(
        f"""
        copy (
            select g.gameId, cast(g.gameDate as date) as gameDate, g.season, g.seasonType, g.seriesId,
                   th.franchiseId as homeFranchiseId, ta.franchiseId as awayFranchiseId,
                   ch.conference as homeConference, ca.conference as awayConference,
                   g.homeScore, g.awayScore, g.homeScore - g.awayScore as margin
            from games g
            join teams th on th.teamId = g.homeTeamId
            join teams ta on ta.teamId = g.awayTeamId
            join conf ch on ch.franchiseId = th.franchiseId
            join conf ca on ca.franchiseId = ta.franchiseId
            where g.seasonYear >= {FIRST_SEASON_YEAR}
            order by g.gameDate, g.gameId
        ) to '{out / "games.parquet"}' (format parquet)
        """
    )
    con.execute(
        f"""
        copy (
            select p.gameId, cast(p.gameDate as date) as gameDate, p.season, t.franchiseId, p.playerId,
                   p.secondsPlayed / 60.0 as minutes, p.plusMinusPoints as plus_minus
            from player_stats p
            join teams t on t.teamId = p.teamId
            join games g on g.gameId = p.gameId
            where g.seasonYear >= {FIRST_SEASON_YEAR} and p.secondsPlayed > 0
            order by p.gameDate, p.gameId, p.playerId
        ) to '{out / "player_minutes.parquet"}' (format parquet)
        """
    )
    con.execute(  # one row per franchise: its latest name, for display
        f"""
        copy (
            select distinct on (franchiseId) franchiseId, teamTricode, teamName
            from teams order by franchiseId, startYear desc
        ) to '{out / "teams.parquet"}' (format parquet)
        """
    )
    return {
        name: con.execute(f"select count(*) from read_parquet('{out / (name + '.parquet')}')").fetchone()[0]
        for name in ("games", "player_minutes")
    }
