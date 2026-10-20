"""Margin model: features, XGBoost fit/predict, Normal probabilities, forward-chaining CV.

The predictive distribution of the home margin is Normal(mu, sigma), with one global sigma.
Margins are integers and never 0, but the Normal is treated as continuous: no continuity correction.
Development never reads 2025-26: `training_rows` asserts it. `fit` has no such guard, so the replay can train on
earlier 2025-26 games.
"""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm
from xgboost import XGBRegressor

from nbaxp.features.build import FEATURES

TRAIN_SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2025)]  # 2015-16 .. 2024-25
VAL_SEASONS = TRAIN_SEASONS[-4:]  # 2021-22 .. 2024-25
TEST_SEASON = "2025-26"

FEATURE_COLS = (
    [f"{side}_{k}" for side in ("home", "away") for k in FEATURES]
    + [f"diff_{k}" for k, has_diff in FEATURES.items() if has_diff]
    + ["is_play_in", "is_playoffs", "is_nba_cup"]
)

BLOWOUT = 15  # |margin| >= 15
TIGHT = 5  # |margin| <= 5
TARGETS = {
    "p_home_win": lambda m: m > 0,
    "p_blowout": lambda m: np.abs(m) >= BLOWOUT,
    "p_tight": lambda m: np.abs(m) <= TIGHT,
}

BASE_PARAMS = {"subsample": 0.8, "colsample_bytree": 0.8, "random_state": 0, "n_jobs": 1}  # 1 thread: deterministic, and no OpenMP oversubscription on a busy machine
GRID = [
    {"max_depth": d, "min_child_weight": w, "n_estimators": n, "learning_rate": 0.03}
    for d, w, n in itertools.product((2, 3, 4), (10, 50), (150, 300))
]


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Seasons 2015-16 to 2024-25, bubble games excluded."""
    out = df[df["season"].isin(TRAIN_SEASONS) & ~df["is_bubble"]].reset_index(drop=True)
    assert TEST_SEASON not in set(out["season"])
    return out


def folds(df: pd.DataFrame):
    """Forward chaining: yield (validation season, train rows, validation rows)."""
    for season in VAL_SEASONS:
        yield season, df[df["season"] < season], df[df["season"] == season]


def fit(train: pd.DataFrame, params: dict) -> XGBRegressor:
    return XGBRegressor(**BASE_PARAMS, **params).fit(train[FEATURE_COLS].astype(float), train["margin"])


def predict_mu(model: XGBRegressor, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[FEATURE_COLS].astype(float))


def probabilities(mu: np.ndarray, sigma: float) -> dict[str, np.ndarray]:
    mu = np.asarray(mu, dtype=float)
    cdf = lambda x: norm.cdf(x, loc=mu, scale=sigma)
    return {
        "p_home_win": 1 - cdf(0),
        "p_blowout": cdf(-BLOWOUT) + 1 - cdf(BLOWOUT),
        "p_tight": cdf(TIGHT) - cdf(-TIGHT),
    }


def home_only(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    return np.full(len(val), train["margin"].mean())


def elo(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """Linear regression of margin on diff_elo; the intercept is the home advantage."""
    slope, intercept = np.polyfit(train["diff_elo"], train["margin"], 1)
    return intercept + slope * val["diff_elo"].to_numpy()


def xgboost(params: dict):
    return lambda train, val: predict_mu(fit(train, params), val)


def cv_predict(df: pd.DataFrame, model) -> pd.DataFrame:
    """Out-of-fold predictions of `model(train, val) -> mu` over the validation seasons."""
    parts = [
        pd.DataFrame({"season": season, "margin": val["margin"].to_numpy(), "mu": model(train, val)})
        for season, train, val in folds(df)
    ]
    return pd.concat(parts, ignore_index=True)


def pooled_sigma(oof: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean((oof["margin"] - oof["mu"]) ** 2)))


def metrics(margin, mu, sigma: float) -> dict[str, float]:
    margin, mu = np.asarray(margin), np.asarray(mu)
    out = {"mae": float(np.mean(np.abs(margin - mu))), "rmse": float(np.sqrt(np.mean((margin - mu) ** 2)))}
    for name, p in probabilities(mu, sigma).items():
        y = TARGETS[name](margin)
        out[f"brier_{name}"] = float(np.mean((p - y) ** 2))
        if name == "p_home_win":
            q = np.clip(p, 1e-15, 1 - 1e-15)
            out["logloss_p_home_win"] = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    return out


def evaluate(oof: pd.DataFrame, sigma: float) -> dict:
    """Pooled and per-fold metrics for one model, all with the same pooled sigma."""
    return {
        "pooled": metrics(oof["margin"], oof["mu"], sigma),
        "folds": {s: metrics(g["margin"], g["mu"], sigma) for s, g in oof.groupby("season")},
    }


def calibration(oof: pd.DataFrame, sigma: float, bins: int = 10) -> dict[str, list[dict]]:
    """Per probability: up to 10 equal-count bins of the predicted value, with mean predicted, observed rate, count."""
    tables = {}
    for name, p in probabilities(oof["mu"], sigma).items():
        y = TARGETS[name](oof["margin"].to_numpy())
        bin_id = pd.qcut(p, bins, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "bin": bin_id}).groupby("bin")
        tables[name] = [
            {"bin": int(b), "mean_predicted": float(t["p"].mean()), "observed": float(t["y"].mean()), "n": len(t)}
            for b, t in g
        ]
    return tables
