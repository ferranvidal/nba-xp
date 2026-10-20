"""Replay tests on a tiny synthetic season: no data dir, no MLflow."""

import numpy as np
import pandas as pd
import pytest

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


def test_promotion_needs_both_metrics():
    assert replay.promote((11.0, 0.21), (10.9, 0.20))
    assert not replay.promote((11.0, 0.21), (10.9, 0.22))  # better MAE, worse Brier
    assert not replay.promote((11.0, 0.21), (11.1, 0.20))  # worse MAE, better Brier
    assert not replay.promote((11.0, 0.21), (11.0, 0.20))  # tie on MAE


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
