"""Model tests on synthetic data: no data dir, no MLflow."""

import numpy as np
import pandas as pd
import pytest

from nbaxp.model import core


def synthetic(seasons, n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        d = pd.DataFrame(rng.normal(size=(n, len(core.FEATURE_COLS))), columns=core.FEATURE_COLS)
        d["season"] = s
        d["is_bubble"] = False
        d["margin"] = (3 * d["diff_elo"] + rng.normal(scale=5, size=n)).round().astype(int)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_probabilities_known_values():
    p = core.probabilities(np.array([0.0, 10.0, -10.0]), 12.0)
    assert p["p_home_win"][0] == pytest.approx(0.5)
    assert p["p_home_win"][1] == pytest.approx(1 - p["p_home_win"][2])
    assert p["p_blowout"][1] == pytest.approx(p["p_blowout"][2])
    assert p["p_tight"][1] == pytest.approx(p["p_tight"][2])
    # sigma = 1, mu = 0: P(|Z| >= 15) ~ 0 and P(|Z| <= 5) ~ 1
    q = core.probabilities(np.array([0.0]), 1.0)
    assert q["p_blowout"][0] < 1e-9 and q["p_tight"][0] > 1 - 1e-6
    # sigma = 10, mu = 0: P(|Z| <= 0.5) = 0.3829
    assert core.probabilities(np.array([0.0]), 10.0)["p_tight"][0] == pytest.approx(0.3829, abs=1e-4)


def test_folds_are_forward_chaining():
    df = synthetic(core.TRAIN_SEASONS + [core.TEST_SEASON], n=3)
    rows = core.training_rows(df)
    assert core.TEST_SEASON not in set(rows["season"])
    seen = [season for season, train, val in core.folds(rows)]
    assert seen == ["2021-22", "2022-23", "2023-24", "2024-25"]
    for season, train, val in core.folds(rows):
        assert set(val["season"]) == {season}
        assert train["season"].max() < season
        assert core.TEST_SEASON not in set(train["season"])


def test_training_rows_drop_bubble_and_test_season():
    df = synthetic(["2019-20", "2025-26"], n=4)
    df.loc[0, "is_bubble"] = True
    assert len(core.training_rows(df)) == 3


def test_end_to_end_fit_predict_and_baselines():
    df = synthetic(core.TRAIN_SEASONS)
    params = {**core.GRID[0], "n_estimators": 30}
    results = {}
    for name, fn in (("home", core.home_only), ("elo", core.elo), ("xgb", core.xgboost(params))):
        oof = core.cv_predict(df, fn)
        assert len(oof) == 4 * 60
        results[name] = core.evaluate(oof, core.pooled_sigma(oof))["pooled"]
    # the signal is linear in diff_elo, so Elo and XGBoost both beat the constant
    assert results["elo"]["mae"] < results["home"]["mae"]
    assert results["xgb"]["mae"] < results["home"]["mae"]
    model = core.fit(df, params)
    mu = core.predict_mu(model, df.head(5))
    assert mu.shape == (5,) and np.isfinite(mu).all()
    assert all(0 <= v.min() and v.max() <= 1 for v in core.probabilities(mu, 5.0).values())
