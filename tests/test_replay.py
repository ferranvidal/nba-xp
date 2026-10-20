"""Replay tests on a tiny synthetic season: no data dir, no MLflow."""

import json

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from nbaxp import replay
from nbaxp.model import core

PARAMS = {"max_depth": 2, "min_child_weight": 1, "n_estimators": 10, "learning_rate": 0.1}
DAY = pd.Timestamp


def synthetic_season(n_days=120, per_day=4, seed=0):
    """2024-25 (training) and 2025-26 (replay) games; margin depends on diff_elo."""
    rng = np.random.default_rng(seed)
    rows = []
    for season, first in (("2024-25", "2024-10-22"), ("2025-26", "2025-10-21")):
        for d in pd.date_range(first, periods=n_days):
            for i in range(per_day):
                rows.append(
                    {"gameId": f"{d:%Y%m%d}{i}", "gameDate": d, "season": season, "seasonType": "Regular Season"}
                )
    df = pd.DataFrame(rows)
    df[core.FEATURE_COLS] = rng.normal(size=(len(df), len(core.FEATURE_COLS)))
    df["homeFranchiseId"], df["awayFranchiseId"] = "1", "2"
    df["is_bubble"] = False
    df["margin"] = (3 * df["diff_elo"] + rng.normal(scale=5, size=len(df))).round().astype(int)
    return df


def run(df, dev_mae=0.0):
    """Replay with a fake registry; return the outputs and the max training date of each fit after the first."""
    train = core.training_rows(df)
    champion = {"model": core.fit(train, PARAMS), "version": "1", "train": train}
    versions = iter("23456789")
    return replay.simulate(df, champion, PARAMS, 13.9, dev_mae, lambda model: next(versions))


def test_retrain_window_excludes_last_30_days(monkeypatch):
    df = synthetic_season()
    fits = []
    real_fit = core.fit
    monkeypatch.setattr(
        core, "fit", lambda train, params: fits.append(train["gameDate"].max()) or real_fit(train, params)
    )
    out = run(df)
    fits = fits[1:]  # the first fit is the starting champion
    checks = out["checks"]
    assert len(checks) > 0
    assert len(fits) == len(checks)
    for fit_max, day in zip(fits, checks["check_date"]):
        assert fit_max < day - pd.Timedelta(days=30)
    # earlier 2025-26 games are used once they are 30 days old
    assert fits[-1] >= pd.Timestamp("2025-10-21")
    day = DAY("2026-01-15")
    rows = replay.retrain_rows(df, day)
    assert rows["gameDate"].max() == day - pd.Timedelta(days=31)
    assert (
        replay.window_rows(df, day)["gameDate"].between(day - pd.Timedelta(days=30), day - pd.Timedelta(days=1)).all()
    )


def test_retrain_rows_drop_bubble_and_early_seasons():
    df = synthetic_season(n_days=5)
    df.loc[df["season"] == "2024-25", "is_bubble"] = True
    assert replay.retrain_rows(df, DAY("2026-06-01"))["season"].eq("2025-26").all()


class Stub:
    """Stands in for a fitted model: `predict` maps the feature frame to mu."""

    def __init__(self, mu_fn):
        self.mu_fn = mu_fn

    def predict(self, X):
        return self.mu_fn(X)


def bootstrap_case(challenger_fn):
    """Games where margin = 3 * diff_elo + noise. The champion predicts 3 * diff_elo + 3 (a bias of 3 points)."""
    rows = synthetic_season(n_days=50)
    rows = rows[rows["season"] == core.TEST_SEASON]
    champion = Stub(lambda X: 3 * X["diff_elo"].to_numpy() + 3)
    return replay.paired_bootstrap(champion, Stub(challenger_fn), rows, 13.9)


def test_bootstrap_promotes_clearly_better_challenger():
    p_mae, p_brier = bootstrap_case(lambda X: 3 * X["diff_elo"].to_numpy())  # removes the bias
    assert p_mae >= 0.9 and p_brier >= 0.9
    assert replay.promote(p_mae, p_brier)


def test_bootstrap_does_not_promote_identical_challenger():
    p_mae, p_brier = bootstrap_case(lambda X: 3 * X["diff_elo"].to_numpy() + 3)
    assert (p_mae, p_brier) == (0.0, 0.0)  # ties never count as a win
    assert not replay.promote(p_mae, p_brier)


def test_bootstrap_does_not_promote_noise_sized_gap():
    noise = np.random.default_rng(1).normal(scale=0.05, size=200)
    p_mae, p_brier = bootstrap_case(lambda X: 3 * X["diff_elo"].to_numpy() + 3 + noise)
    assert not replay.promote(p_mae, p_brier)
    assert 0.1 < p_mae < 0.9  # a coin flip, far from either end


def test_bootstrap_is_deterministic():
    fn = lambda X: 3 * X["diff_elo"].to_numpy() + 0.5
    assert bootstrap_case(fn) == bootstrap_case(fn)


def test_promotion_needs_both_metrics():
    assert replay.promote(0.90, 0.95)
    assert not replay.promote(0.89, 0.99)
    assert not replay.promote(0.99, 0.89)


def test_drift_skips_game_type_flags():
    assert not set(replay.DRIFT_COLS) & set(replay.GAME_TYPE_FLAGS)
    assert len(replay.DRIFT_COLS) == len(core.FEATURE_COLS) - 3  # is_play_in, is_playoffs, is_nba_cup


def test_seasonal_reference_matches_window_days_into_season():
    df = synthetic_season()  # both seasons have 120 days; 2025-26 starts 2025-10-21, 2024-25 on 2024-10-22
    test = df[df["season"] == core.TEST_SEASON]
    win = replay.window_rows(test, DAY("2025-12-01"))  # days 11..40 of the season
    ref = replay.seasonal_reference(core.training_rows(df), test, win)
    assert ref["season"].eq("2024-25").all()
    assert ref["gameDate"].min() == DAY("2024-11-02") and ref["gameDate"].max() == DAY("2024-12-01")
    assert len(ref) == 30 * 4


def test_export_serving_round_trips(tmp_path):
    df = synthetic_season(n_days=30)
    model = core.fit(core.training_rows(df), PARAMS)
    replay.export_serving(model, {"sigma": 13.9, "model_version": "3"}, tmp_path / "serving")
    meta = json.loads((tmp_path / "serving" / "meta.json").read_text())
    assert meta["sigma"] == 13.9 and meta["feature_cols"] == core.FEATURE_COLS
    loaded = XGBRegressor()
    loaded.load_model(tmp_path / "serving" / "model.json")
    np.testing.assert_array_equal(core.predict_mu(loaded, df), core.predict_mu(model, df))


def test_trigger_rules():
    start, dev = DAY("2025-10-21"), 10.9
    assert replay.trigger(DAY("2025-11-19"), DAY("2025-11-18"), start, None, 20.0, dev) is None  # 29 days
    assert replay.trigger(DAY("2025-11-20"), DAY("2025-11-19"), start, None, 20.0, dev) == "mae_drift"  # 30 days
    assert replay.trigger(DAY("2025-12-01"), DAY("2025-11-30"), start, None, 10.0, dev) == "monthly"
    assert replay.trigger(DAY("2025-12-02"), DAY("2025-12-01"), start, None, 10.0, dev) is None
    assert replay.trigger(DAY("2025-12-02"), DAY("2025-12-01"), start, None, 12.0, dev) == "mae_drift"
    assert replay.trigger(DAY("2025-12-02"), DAY("2025-12-01"), start, None, 11.9, dev) is None  # +1.0 is not above
    last = DAY("2025-12-01")
    assert replay.trigger(DAY("2025-12-07"), DAY("2025-12-06"), start, last, 15.0, dev) is None  # < 7 days
    assert replay.trigger(DAY("2025-12-08"), DAY("2025-12-07"), start, last, 15.0, dev) == "mae_drift"


def test_psi():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert replay.psi(x, x) == 0
    assert replay.psi(x, rng.normal(size=5000)) < 0.05
    assert replay.psi(x, rng.normal(loc=2, size=5000)) > 1
    assert replay.psi(x, np.where(rng.random(5000) < 0.5, np.nan, x)) > 0.2  # NaN is its own bin
    assert replay.psi(np.zeros(100), np.zeros(10)) == 0  # constant feature
    flag = np.r_[np.zeros(95), np.ones(5)]
    assert replay.psi(flag, flag) == 0 and replay.psi(flag, np.ones(50)) > 1  # binary feature


def test_replay_is_deterministic_and_complete():
    df = synthetic_season()
    a, b = run(df), run(df)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])
    pred = a["predictions"]
    assert len(pred) == (df["season"] == core.TEST_SEASON).sum()
    assert pred["gameDate"].is_monotonic_increasing
    assert set(pred["model_version"]) <= {"1", *"23456789"}
    assert pred[["p_home_win", "p_blowout", "p_tight"]].apply(lambda c: c.between(0, 1).all()).all()
    assert (pred["sigma"] == 13.9).all()
    assert len(a["daily"]) == pred["gameDate"].nunique()
    assert pytest.approx(a["daily"]["rolling_mae_30d"].iloc[-1]) == (pred["margin"] - pred["mu"]).abs().tail(120).mean()
