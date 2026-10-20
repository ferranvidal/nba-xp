"""API tests on a tiny fixture: a small model, six games, two teams. No real data."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nbaxp import api
from nbaxp.model import core

SIGMA = 13.9


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "gameId": ["0022400001", "0022500001", "0022500002", "0022500003", "0022500004", "0022500005"],
            "gameDate": pd.to_datetime(["2025-04-01"] + ["2025-10-21"] * 2 + ["2025-10-22"] * 3),
            "season": ["2024-25"] + [core.TEST_SEASON] * 5,
            "homeFranchiseId": "1",
            "awayFranchiseId": "2",
            "margin": [3, 10, -7, 1, 25, -12],
        }
    )
    df[core.FEATURE_COLS] = rng.normal(size=(len(df), len(core.FEATURE_COLS)))
    model = core.fit(df, {"max_depth": 2, "min_child_weight": 1, "n_estimators": 5, "learning_rate": 0.1})
    (root / "serving").mkdir()
    model.save_model(root / "serving" / "model.json")
    meta = {"sigma": SIGMA, "feature_cols": core.FEATURE_COLS, "model_version": "7", "train_cutoff": "2026-04-01"}
    (root / "serving" / "meta.json").write_text(json.dumps(meta))
    (root / "features").mkdir()
    df.to_parquet(root / "features" / "features.parquet")
    (root / "staged").mkdir()
    pd.DataFrame(
        {"franchiseId": ["1", "2"], "teamTricode": ["BOS", "NYK"], "teamName": ["Boston Celtics", "New York Knicks"]}
    ).to_parquet(root / "staged" / "teams.parquet")
    client = TestClient(api.create_app(root))
    client.mu = core.predict_mu(model, df[df["season"] == core.TEST_SEASON])
    return client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "7", "train_cutoff": "2026-04-01"}


def test_games_on_a_date(client):
    games = client.get("/games/2025-10-22").json()
    assert [g["gameId"] for g in games] == ["0022500003", "0022500004", "0022500005"]
    g = games[1]
    assert g["home"] == {"tricode": "BOS", "name": "Boston Celtics"} and g["away"]["tricode"] == "NYK"
    assert g["margin"] == 25 and g["sigma"] == SIGMA
    assert g["mu"] == pytest.approx(client.mu[3], abs=1e-5)
    expected = core.probabilities(np.array([g["mu"]]), SIGMA)
    for name, p in expected.items():
        assert g[name] == pytest.approx(p[0])


def test_games_date_errors(client):
    assert client.get("/games/2025-10-23").status_code == 404
    assert client.get("/games/2025-04-01").status_code == 404  # 2024-25 is not served
    assert client.get("/games/not-a-date").status_code == 422


def test_predict(client):
    r = client.post("/predict", json={"game_id": "0022500002"})
    assert r.status_code == 200
    assert r.json() == client.get("/games/2025-10-21").json()[1]
    assert client.post("/predict", json={"game_id": "0022400001"}).status_code == 404  # 2024-25 is not served
    assert client.post("/predict", json={"game_id": "nope"}).status_code == 404
    assert client.post("/predict", json={}).status_code == 422
