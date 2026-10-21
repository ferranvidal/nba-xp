"""`nbaxp train`: CV, MLflow tracking, registry, and the dev report. Also `load_champion` for Phase 4."""

import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from nbaxp.model import core

MODEL_NAME = "nbaxp-margin"
CHAMPION = "champion"
EXPERIMENT = "nbaxp-margin"


def setup_mlflow(data_dir: Path) -> None:
    root = (data_dir / "mlflow").resolve()
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    mlflow.set_registry_uri(f"sqlite:///{root / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(root / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def mean_fold_mae(result: dict) -> float:
    return sum(f["mae"] for f in result["folds"].values()) / len(result["folds"])


def champion_features(name: str = MODEL_NAME) -> list[str] | None:
    """Feature list of the champion, or None if there is no champion."""
    try:
        return mlflow.models.get_model_info(f"models:/{name}@{CHAMPION}").metadata["feature_cols"]
    except MlflowException:
        return None


def register(model, sigma: float, name: str = MODEL_NAME) -> str:
    """Log the model with sigma and the feature list, register a new version. Set the alias if there is no champion,
    or if the champion was trained on another feature list: it cannot score the current feature table."""
    stale = champion_features(name) != core.FEATURE_COLS
    info = mlflow.xgboost.log_model(
        model,
        name="model",
        registered_model_name=name,
        metadata={"sigma": sigma, "feature_cols": core.FEATURE_COLS},
    )
    if stale:
        MlflowClient().set_registered_model_alias(name, CHAMPION, info.registered_model_version)
    return info.registered_model_version


def train(data_dir: Path = Path("data")) -> dict:
    df = core.training_rows(pd.read_parquet(data_dir / "features" / "features.parquet"))
    setup_mlflow(data_dir)

    with mlflow.start_run(run_name="train"):
        mlflow.log_params({"train_seasons": f"{core.TRAIN_SEASONS[0]}..{core.TRAIN_SEASONS[-1]}", "n_rows": len(df)})
        models = {}
        for name, fn in (("home_only", core.home_only), ("elo", core.elo)):
            oof = core.cv_predict(df, fn)
            sigma = core.pooled_sigma(oof)
            models[name] = {"sigma": sigma, **core.evaluate(oof, sigma), "calibration": core.calibration(oof, sigma)}

        grid = []
        for params in core.GRID:
            with mlflow.start_run(run_name=str(params), nested=True):
                oof = core.cv_predict(df, core.xgboost(params))
                sigma = core.pooled_sigma(oof)
                res = {"params": params, "sigma": sigma, **core.evaluate(oof, sigma)}
                res["mean_val_mae"] = mean_fold_mae(res)
                res["oof"] = oof
                mlflow.log_params(params)
                mlflow.log_metrics({"mean_val_mae": res["mean_val_mae"], "sigma": sigma, **res["pooled"]})
                for season, m in res["folds"].items():
                    mlflow.log_metric("val_mae", m["mae"], step=int(season[:4]))
                grid.append(res)

        best = min(grid, key=lambda r: r["mean_val_mae"])
        models["xgboost"] = {
            "sigma": best["sigma"],
            "pooled": best["pooled"],
            "folds": best["folds"],
            "calibration": core.calibration(best["oof"], best["sigma"]),
        }

        final = core.fit(df, best["params"])
        gain = final.get_booster().get_score(importance_type="gain")
        importance = dict(sorted(gain.items(), key=lambda kv: -kv[1]))

        report = {
            "val_seasons": core.VAL_SEASONS,
            "n_train_rows": len(df),
            "selected_params": best["params"],
            "mean_val_mae_selected": best["mean_val_mae"],
            "sigma": best["sigma"],
            "feature_cols": core.FEATURE_COLS,
            "models": models,
            "grid": [
                {"params": r["params"], "mean_val_mae": r["mean_val_mae"], "sigma": r["sigma"]} for r in grid
            ],
            "feature_importance_gain": importance,
        }
        out = data_dir / "reports" / "dev_cv.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))

        mlflow.log_params({f"selected_{k}": v for k, v in best["params"].items()})
        mlflow.log_param("sigma", best["sigma"])
        mlflow.log_metrics({f"pooled_{k}": v for k, v in best["pooled"].items()})
        mlflow.log_artifact(str(out))
        rows = [
            {"model": m, "target": t, **row}
            for m, v in models.items()
            for t, table in v["calibration"].items()
            for row in table
        ]
        mlflow.log_table(pd.DataFrame(rows), "calibration.json")
        report["registered_version"] = register(final, best["sigma"])
    return report


def has_champion(data_dir: Path = Path("data")) -> bool:
    """True if a champion exists and was trained on the current feature list."""
    setup_mlflow(data_dir)
    return champion_features() == core.FEATURE_COLS


def load_champion(data_dir: Path = Path("data")):
    """Return (model, sigma, feature_cols) of the model with alias `champion`."""
    setup_mlflow(data_dir)
    uri = f"models:/{MODEL_NAME}@{CHAMPION}"
    meta = mlflow.models.get_model_info(uri).metadata
    return mlflow.xgboost.load_model(uri), meta["sigma"], meta["feature_cols"]
