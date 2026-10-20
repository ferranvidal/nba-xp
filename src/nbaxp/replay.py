"""`nbaxp replay`: replay the 2025-26 season day by day as simulated production.

For each game day D, in date order: (1) retrain check if a rule fires, (2) predict D's games with the current
champion and the as-of features, (3) receive D's results and evaluate. The check runs before the predictions of D,
so a promoted challenger predicts D. A check on D uses only games before D.

Sigma stays fixed at the dev value for every model version. Hyperparameters stay fixed at the dev values.
The replay registers its own model `nbaxp-margin-replay`: deleted and recreated at start, v1 = copy of the dev
champion. A rerun from scratch gives identical outputs.
"""

import json
from collections.abc import Callable
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from nbaxp.model import core
from nbaxp.model.tracking import CHAMPION, load_champion, register, setup_mlflow

REPLAY_MODEL = "nbaxp-margin-replay"
WINDOW = pd.Timedelta(days=30)  # training cutoff, scoring window, rolling MAE and drift window
MIN_GAP = pd.Timedelta(days=7)  # at most one MAE-triggered check per 7 days
MAE_MARGIN = 1.0  # rolling MAE above dev MAE + 1.0 fires a check
PSI_FLAG = 0.2
PSI_EPS = 1e-4  # floor for a bin share, so empty bins do not give log(0)
CHECK_COLS = [
    "check_date", "trigger", "rolling_mae", "n_window_games", "n_train_rows",
    "champion_version", "champion_mae", "champion_brier", "challenger_mae", "challenger_brier",
    "promoted", "new_version", "n_drift_flags",
]  # fmt: skip


def retrain_rows(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Games from 2015-16 on, bubble excluded, played before `day` - 30 days."""
    return df[(df["season"] >= core.TRAIN_SEASONS[0]) & ~df["is_bubble"] & (df["gameDate"] < day - WINDOW)]


def window_rows(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Games in [day - 30 days, day)."""
    return df[(df["gameDate"] >= day - WINDOW) & (df["gameDate"] < day)]


def psi(expected, actual) -> float:
    """Population stability index. Bins: the distinct decile edges of `expected`, a tail bin above the last
    edge, and one bin for NaN. Every bin share has a floor of 1e-4."""
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, 11))) if (~np.isnan(expected)).any() else np.array([])

    def shares(x):
        bins = np.where(np.isnan(x), len(edges) + 1, np.searchsorted(edges, x))
        return np.clip(np.bincount(bins, minlength=len(edges) + 2) / len(x), PSI_EPS, None)

    e, a = shares(expected), shares(actual)
    return float(np.sum((a - e) * np.log(a / e)))


def trigger(day, prev_day, start, last_check, rolling_mae: float, dev_mae: float) -> str | None:
    """Name of the rule that fires on `day`, or None. No check before 30 days of replay results exist."""
    if day - start < WINDOW:
        return None
    if prev_day is not None and day.to_period("M") != prev_day.to_period("M"):
        return "monthly"
    if (last_check is None or day - last_check >= MIN_GAP) and rolling_mae > dev_mae + MAE_MARGIN:
        return "mae_drift"
    return None


def promote(champion: tuple[float, float], challenger: tuple[float, float]) -> bool:
    """Scores are (MAE, Brier of p_home_win). The challenger must be lower on both."""
    return challenger[0] < champion[0] and challenger[1] < champion[1]


def score(model, rows: pd.DataFrame, sigma: float) -> tuple[float, float]:
    m = core.metrics(rows["margin"], core.predict_mu(model, rows), sigma)
    return m["mae"], m["brier_p_home_win"]


def retrain_check(df, test, champion, day, why, rolling_mae, params, sigma, register_model):
    """One check: drift on the last 30 days, challenger vs champion on the same days. Returns (champion, row, drift)."""
    win = window_rows(test, day)
    drift = [
        {"check_date": day, "feature": c, "psi": psi(champion["train"][c], win[c])} for c in core.FEATURE_COLS
    ]  # record-count features (wins, losses) drift by construction early in a season; so do the game-type flags
    for d in drift:
        d["flagged"] = d["psi"] > PSI_FLAG
    train = retrain_rows(df, day)
    challenger = core.fit(train, params)
    champ_score, chall_score = score(champion["model"], win, sigma), score(challenger, win, sigma)
    promoted = promote(champ_score, chall_score)
    row = {
        "check_date": day, "trigger": why, "rolling_mae": rolling_mae, "n_window_games": len(win),
        "n_train_rows": len(train), "champion_version": champion["version"],
        "champion_mae": champ_score[0], "champion_brier": champ_score[1],
        "challenger_mae": chall_score[0], "challenger_brier": chall_score[1],
        "promoted": promoted, "new_version": None, "n_drift_flags": sum(d["flagged"] for d in drift),
    }  # fmt: skip
    if promoted:
        row["new_version"] = register_model(challenger)
        champion = {"model": challenger, "version": row["new_version"], "train": train}
    return champion, row, drift


def simulate(df, champion, params, sigma, dev_mae, register_model: Callable) -> dict[str, pd.DataFrame]:
    """The replay loop. `champion` = {"model", "version", "train"}; `register_model(model) -> version`."""
    test = df[df["season"] == core.TEST_SEASON].sort_values(["gameDate", "gameId"])
    start = test["gameDate"].min()
    parts, daily, checks, drift = [], [], [], []
    prev_day = last_check = None
    for day, rows in test.groupby("gameDate"):
        done = pd.concat(parts) if parts else test.iloc[:0].assign(mu=[], model_version=[])
        win = window_rows(done, day)
        rolling = float(np.mean(np.abs(win["margin"] - win["mu"]))) if len(win) else np.nan
        why = trigger(day, prev_day, start, last_check, rolling, dev_mae)
        if why:
            champion, row, d = retrain_check(df, test, champion, day, why, rolling, params, sigma, register_model)
            checks.append(row)
            drift += d
            last_check = day
        mu = core.predict_mu(champion["model"], rows)
        out = rows[["gameId", "gameDate", "seasonType", "homeFranchiseId", "awayFranchiseId", "margin"]].assign(
            mu=mu, sigma=sigma, **core.probabilities(mu, sigma), model_version=champion["version"]
        )
        parts.append(out)
        m = core.metrics(out["margin"], mu, sigma)
        recent = pd.concat(parts)
        recent = recent[recent["gameDate"] > day - WINDOW]
        daily.append(
            {
                "gameDate": day,
                "n_games": len(out),
                "mae": m["mae"],
                "brier_p_home_win": m["brier_p_home_win"],
                "brier_p_blowout": m["brier_p_blowout"],
                "brier_p_tight": m["brier_p_tight"],
                "rolling_mae_30d": float(np.mean(np.abs(recent["margin"] - recent["mu"]))),
            }
        )
        prev_day = day
    return {
        "predictions": pd.concat(parts, ignore_index=True),
        "daily": pd.DataFrame(daily),
        "checks": pd.DataFrame(checks, columns=CHECK_COLS),
        "drift": pd.DataFrame(drift, columns=["check_date", "feature", "psi", "flagged"]),
    }


def mae_by_type(frame: pd.DataFrame, mu) -> dict:
    err = pd.DataFrame({"t": frame["seasonType"].to_numpy(), "e": np.abs(frame["margin"].to_numpy() - mu)})
    return {t: {"n": len(g), "mae": float(g["e"].mean())} for t, g in err.groupby("t")}


def replay(data_dir: Path = Path("data")) -> dict:
    df = pd.read_parquet(data_dir / "features" / "features.parquet")
    dev = json.loads((data_dir / "reports" / "dev_cv.json").read_text())
    params, dev_mae = dev["selected_params"], dev["models"]["xgboost"]["pooled"]["mae"]

    setup_mlflow(data_dir)
    dev_model, sigma, _ = load_champion(data_dir)
    client = MlflowClient()
    try:
        client.delete_registered_model(REPLAY_MODEL)
    except MlflowException:
        pass  # first run: nothing to delete
    for run in mlflow.search_runs(filter_string="attributes.run_name = 'replay'", output_format="list"):
        client.delete_run(run.info.run_id)

    def register_model(model) -> str:
        version = register(model, sigma, REPLAY_MODEL)
        client.set_registered_model_alias(REPLAY_MODEL, CHAMPION, version)
        return str(version)

    with mlflow.start_run(run_name="replay"):
        mlflow.log_params({"sigma": sigma, "dev_mae": dev_mae, **{f"param_{k}": v for k, v in params.items()}})
        champion = {"model": dev_model, "version": register_model(dev_model), "train": core.training_rows(df)}
        out = simulate(df, champion, params, sigma, dev_mae, register_model)
        for i, row in out["checks"].iterrows():
            mlflow.log_metrics(
                {k: float(row[k]) for k in ("rolling_mae", "champion_mae", "champion_brier", "challenger_mae",
                                            "challenger_brier", "promoted", "n_drift_flags")},
                step=i,
            )  # fmt: skip
        mlflow.log_table(out["checks"].astype({"check_date": str}), "checks.json")

        pred = out["predictions"]
        rows = df.set_index("gameId").loc[pred["gameId"]]  # same order as pred
        frozen_mu = core.predict_mu(dev_model, rows)
        elo_mu = core.elo(core.training_rows(df), rows)
        elo_sigma = dev["models"]["elo"]["sigma"]
        checks, drift = out["checks"], out["drift"]
        summary = {
            "n_games": len(pred),
            "n_game_days": len(out["daily"]),
            "sigma": sigma,
            "dev_mae": dev_mae,
            "champion_chain": core.metrics(pred["margin"], pred["mu"], sigma),
            "frozen_v1": core.metrics(pred["margin"], frozen_mu, sigma),
            "elo_baseline": core.metrics(pred["margin"], elo_mu, elo_sigma),
            "mae_by_season_type": {
                "champion_chain": mae_by_type(pred, pred["mu"].to_numpy()),
                "frozen_v1": mae_by_type(pred, frozen_mu),
                "elo_baseline": mae_by_type(pred, elo_mu),
            },
            "n_checks": len(checks),
            "n_promoted": int(checks["promoted"].sum()),
            "checks_by_trigger": checks.groupby("trigger")["promoted"].agg(n="size", promoted="sum").astype(int)
            .to_dict("index"),
            "versions_used": pred["model_version"].value_counts().sort_index().to_dict(),
            "drift_flags": {  # feature -> check dates with PSI > 0.2
                f: [str(d.date()) for d in g["check_date"]] for f, g in drift[drift["flagged"]].groupby("feature")
            },
        }  # fmt: skip

        out_dir = data_dir / "replay"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in out.items():
            frame.to_parquet(out_dir / f"{name}.parquet", index=False)
        report = data_dir / "reports" / "replay.json"
        report.write_text(json.dumps(summary, indent=2))
        mlflow.log_metrics({f"chain_{k}": v for k, v in summary["champion_chain"].items()})
        mlflow.log_metrics({f"frozen_{k}": v for k, v in summary["frozen_v1"].items()})
        mlflow.log_artifacts(str(out_dir), "replay")
        mlflow.log_artifact(str(report))
    return summary
