"""`nbaxp api`: serve the model's predictions for the real 2025-26 games.

Reads `serving/` (the final champion of the replay: model.json, meta.json), `features/features.parquet` (the
as-of feature row of each game) and `staged/teams.parquet` (team names). Predicts every 2025-26 game once at startup.
A game before the training cutoff in `serving/meta.json` was in the model's training set.
"""

import datetime
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBRegressor

from nbaxp.model import core


class Team(BaseModel):
    tricode: str
    name: str


class Game(BaseModel):
    gameId: str
    gameDate: datetime.date
    home: Team
    away: Team
    mu: float  # predicted home margin
    sigma: float
    p_home_win: float
    p_blowout: float
    p_tight: float
    margin: int  # actual home margin


class PredictRequest(BaseModel):
    game_id: str


def create_app(data_dir: Path = Path("data")) -> FastAPI:
    meta = json.loads((data_dir / "serving" / "meta.json").read_text())
    assert meta["feature_cols"] == core.FEATURE_COLS, "serving model was trained on other features"
    model = XGBRegressor()
    model.load_model(data_dir / "serving" / "model.json")

    df = pd.read_parquet(data_dir / "features" / "features.parquet")
    df = df[df["season"] == core.TEST_SEASON].reset_index(drop=True)
    teams = pd.read_parquet(data_dir / "staged" / "teams.parquet").set_index("franchiseId")
    mu = core.predict_mu(model, df).tolist()
    probs = {k: v.tolist() for k, v in core.probabilities(mu, meta["sigma"]).items()}
    team = lambda franchise: Team(tricode=teams.at[franchise, "teamTricode"], name=teams.at[franchise, "teamName"])
    games = {
        row.gameId: Game(
            gameId=row.gameId, gameDate=row.gameDate.date(), home=team(row.homeFranchiseId),
            away=team(row.awayFranchiseId), mu=mu[i], sigma=meta["sigma"], margin=int(row.margin),
            **{k: v[i] for k, v in probs.items()},
        )
        for i, row in enumerate(df.itertuples())
    }  # fmt: skip

    app = FastAPI(title="nbaxp")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": meta["model_version"], "train_cutoff": meta["train_cutoff"]}

    @app.get("/games/{date}")
    def games_on(date: datetime.date) -> list[Game]:
        found = [g for g in games.values() if g.gameDate == date]
        if not found:
            raise HTTPException(404, f"no 2025-26 game on {date}")
        return found

    @app.post("/predict")
    def predict(body: PredictRequest) -> Game:
        if body.game_id not in games:
            raise HTTPException(404, f"unknown game {body.game_id}")
        return games[body.game_id]

    return app
