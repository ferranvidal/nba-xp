"""Build the small JSON files that the static site in docs/ draws its charts from.

Usage:  .venv/bin/python scripts/build_site.py

Reads the local pipeline outputs (data/features, data/reports, data/replay, data/mlflow, data/raw) and writes
docs/data/*.json. Nothing here writes to data/ or to the MLflow registry. The output is deterministic: no
timestamps, fixed bootstrap seeds, one XGBoost thread.

Two things are recomputed because the pipeline does not save them:
  - out-of-fold predictions of the development folds (home-only, Elo baseline, XGBoost), with the same code as
    `nbaxp train` (nbaxp.model.core.cv_predict);
  - per-game predictions of the frozen v1 model (the dev champion) and of the Elo baseline on the test season.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from xgboost import XGBRegressor

from nbaxp.model import core
from nbaxp.model.tracking import load_champion

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "docs" / "data"
SEED = 0
BOOT = 2000

# (id, name, feature keys). A key k stands for the columns home_k, away_k and diff_k; "is_*" keys stand for themselves.
FAMILIES = [
    ("elo", "Elo rating", ["elo"]),
    ("record", "Record", ["wins", "losses", "win_pct"]),
    ("streak", "Streak", ["streak"]),
    ("pd", "Recent point differential", ["pd_last5", "pd_last10"]),
    ("rest", "Rest and back-to-back", ["rest_days", "back_to_back"]),
    ("travel", "Travel", ["travel_km_3d", "travel_km_7d"]),
    ("load", "Minute load", ["rotation_min_3d", "rotation_min_7d", "rotation_min_14d"]),
    ("players", "Player-based team strength", ["player_strength", "availability"]),
    ("eliminated", "Eliminated", ["eliminated"]),
    ("win_or_go_home", "Win or go home", ["win_or_go_home"]),
    ("game_type", "Game type", ["is_play_in", "is_playoffs", "is_nba_cup"]),
]
KEY_TO_FAMILY = {k: fid for fid, _, keys in FAMILIES for k in keys}
FAMILY_NAME = {fid: name for fid, name, _ in FAMILIES}


# ---------------------------------------------------------------------------------------------------- helpers
def clean(o, nd=5):
    """Make an object JSON-safe: numpy -> python, NaN -> None, floats rounded."""
    if isinstance(o, dict):
        return {str(k): clean(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, nd) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else round(float(o), nd)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if o is None or (o is pd.NA) or (o is pd.NaT):
        return None
    return o


def write(name: str, obj, nd=5) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(clean(obj, nd), separators=(",", ":"), ensure_ascii=False))


def family_of(col: str) -> str:
    key = col.split("_", 1)[1] if col.split("_", 1)[0] in ("home", "away", "diff") else col
    return KEY_TO_FAMILY[key]


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def calib_tables(margin, mu, sigma: float) -> dict:
    """`core.calibration` (10 equal-count bins) plus a 95% Wilson interval on the observed rate."""
    tables = core.calibration(pd.DataFrame({"margin": np.asarray(margin), "mu": np.asarray(mu)}), sigma)
    for rows in tables.values():
        for r in rows:
            lo, hi = wilson(np.array([r["observed"] * r["n"]]), np.array([r["n"]]))
            r["lo"], r["hi"] = float(lo[0]), float(hi[0])
    return tables


def boot_diff(a: np.ndarray, b: np.ndarray, seed: int = SEED) -> dict:
    """Paired bootstrap of mean(a - b) over games: the mean, a 95% percentile interval, and P(mean < 0)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "p_below_zero": float((means < 0).mean())}


def binned(x: pd.Series, y: pd.Series, kind: str, clip=None, q: int = 10) -> list[dict]:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if kind == "integer":
        lo, hi = clip
        d["g"] = d["x"].clip(lo, hi)
    else:
        d["g"] = pd.qcut(d["x"], q, labels=False, duplicates="drop")
    rows = []
    for _, t in d.groupby("g", sort=True):
        n = len(t)
        row = {"x": float(t["x"].mean()), "mean": float(t["y"].mean()), "se": float(t["y"].std(ddof=1) / math.sqrt(n)), "n": n}
        if kind == "integer":
            v = int(t["g"].iloc[0])
            row["label"] = f"≤{lo}" if v == lo else f"≥{hi}" if v == hi else str(v)
            row["x"] = float(v)
        else:
            row["label"] = f"{t['x'].min():.3g} to {t['x'].max():.3g}"
            row["lo"], row["hi"] = float(t["x"].min()), float(t["x"].max())
        rows.append(row)
    return rows


def categorical(y: pd.Series, groups: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for label, mask in groups.items():
        t = y[mask]
        if len(t) == 0:
            continue
        rows.append({"label": label, "mean": float(t.mean()), "se": float(t.std(ddof=1) / math.sqrt(len(t))) if len(t) > 1 else None, "n": len(t)})
    return rows


def pair_groups(df: pd.DataFrame, home: str, away: str) -> dict[str, pd.Series]:
    h, a = df[home].astype(bool), df[away].astype(bool)
    return {"neither": ~h & ~a, "home only": h & ~a, "away only": ~h & a, "both": h & a}


# ---------------------------------------------------------------------------------------------------- sections
def timeline(feat: pd.DataFrame, tr: pd.DataFrame) -> dict:
    seasons = []
    for s, g in feat.groupby("season"):
        seasons.append({"season": s, "n": len(g), "n_bubble": int(g["is_bubble"].sum()), "n_train": int((tr["season"] == s).sum())})
    return {
        "seasons": seasons,
        "folds": [
            {"val": s, "n_train": len(t), "n_val": len(v), "train_first": core.TRAIN_SEASONS[0],
             "train_last": core.TRAIN_SEASONS[core.TRAIN_SEASONS.index(s) - 1]}
            for s, t, v in core.folds(tr)
        ],
        "final_n_train": len(tr),
        "warmup": sorted(set(feat["season"]) - set(core.TRAIN_SEASONS) - {core.TEST_SEASON}),
        "test": core.TEST_SEASON,
        "bubble_start": "2020-07-01",
    }


def feature_section(tr: pd.DataFrame, dev: dict) -> dict:
    y = tr["margin"].astype(float)
    fams = []

    def panel(key, label, kind, bins, col=None, **extra):
        return {"key": key, "label": label, "kind": kind, "bins": bins, "col": col, **extra}

    num = lambda col, label, **kw: panel(col, label, "numeric", binned(tr[col], y, "numeric"), col=col, **kw)
    intg = lambda col, label, clip: panel(col, label, "integer", binned(tr[col], y, "integer", clip), col=col)
    panels = {
        "elo": [num("diff_elo", "Home Elo rating minus away Elo rating (rating points)")],
        "record": [num("diff_win_pct", "Home win share minus away win share")],
        "streak": [intg("diff_streak", "Home streak minus away streak (games)", (-8, 8))],
        "pd": [num("diff_pd_last10", "Home minus away mean margin, last 10 games (points)")],
        "rest": [
            intg("diff_rest_days", "Home rest days minus away rest days", (-4, 4)),
            panel("back_to_back", "Which team plays a back-to-back", "category",
                  categorical(y, pair_groups(tr, "home_back_to_back", "away_back_to_back"))),
        ],
        "travel": [num("diff_travel_km_7d", "Home minus away km travelled in the last 7 days")],
        "load": [num("diff_rotation_min_7d", "Home minus away rotation minutes in the last 7 days")],
        "players": [num("diff_player_strength", "Home minus away player-based team strength (points per game)")],
        "eliminated": [panel("eliminated", "Which team is eliminated", "category",
                             categorical(y, pair_groups(tr, "home_eliminated", "away_eliminated")))],
        "win_or_go_home": [panel("win_or_go_home", "Which team plays a win-or-go-home game", "category",
                                 categorical(y, pair_groups(tr, "home_win_or_go_home", "away_win_or_go_home")))],
        "game_type": [panel("game_type", "Game type", "category",
                            categorical(y, {t: tr["seasonType"] == t for t in ("Regular Season", "Play-In", "Playoffs", "NBA Cup")}))],
    }
    gain = dev["feature_importance_gain"]
    total_gain = sum(gain.values())
    for fid, name, keys in FAMILIES:
        cols = [c for c in dev["feature_cols"] if family_of(c) == fid]
        fams.append({
            "id": fid, "name": name, "panels": panels[fid],
            "columns": [{"col": c, "r": float(tr[c].astype(float).corr(y)), "gain": float(gain.get(c, 0.0)),
                         "n_missing": int(tr[c].isna().sum())} for c in cols],
            "gain_share": sum(gain.get(c, 0.0) for c in cols) / total_gain,
        })
    top = [{"col": c, "family": family_of(c), "gain": g, "share": g / total_gain} for c, g in list(gain.items())
           ]
    return {
        "n_rows": len(tr), "mean_margin": float(y.mean()), "sd_margin": float(y.std()),
        "families": fams, "importance": top, "n_feature_cols": len(dev["feature_cols"]),
        "n_used": len(gain),
    }


def cv_subset(tr, cols, params):
    def model(train, val):
        m = XGBRegressor(**core.BASE_PARAMS, **params).fit(train[cols].astype(float), train["margin"])
        return m.predict(val[cols].astype(float))

    return core.cv_predict(tr, model)


def ablation(tr, params, full_oof) -> list[dict]:
    cols_of = lambda fids: [c for c in core.FEATURE_COLS if family_of(c) in fids]
    all_ids = [f[0] for f in FAMILIES]
    runs = [(f, "Without " + FAMILY_NAME[f].lower().replace("elo", "Elo"), [f]) for f in all_ids]
    runs += [("elo+record", "Without Elo rating and record", ["elo", "record"])]
    full_err = np.abs(full_oof["margin"] - full_oof["mu"]).to_numpy()
    sigma_full = core.pooled_sigma(full_oof)
    full_m = core.metrics(full_oof["margin"], full_oof["mu"], sigma_full)
    out = [{"id": "all", "name": "All features", "n_cols": len(core.FEATURE_COLS), "mae": full_m["mae"], "brier": full_m["brier_p_home_win"]}]
    for fid, name, drop in runs:
        keep = [c for c in core.FEATURE_COLS if c not in cols_of(drop)]
        oof = cv_subset(tr, keep, params)
        m = core.metrics(oof["margin"], oof["mu"], core.pooled_sigma(oof))
        d = boot_diff(np.abs(oof["margin"] - oof["mu"]).to_numpy(), full_err)
        out.append({"id": fid, "name": name, "n_cols": len(keep), "mae": m["mae"], "brier": m["brier_p_home_win"],
                    "d_mae": d["mean"], "d_lo": d["lo"], "d_hi": d["hi"]})
    keep = cols_of(["elo", "record"])
    oof = cv_subset(tr, keep, params)
    m = core.metrics(oof["margin"], oof["mu"], core.pooled_sigma(oof))
    d = boot_diff(np.abs(oof["margin"] - oof["mu"]).to_numpy(), full_err)
    out.append({"id": "only elo+record", "name": "Only Elo rating and record", "n_cols": len(keep), "mae": m["mae"], "brier": m["brier_p_home_win"],
                "d_mae": d["mean"], "d_lo": d["lo"], "d_hi": d["hi"]})
    return out


def dev_section(tr: pd.DataFrame, dev: dict) -> dict:
    params = dev["selected_params"]
    val = pd.concat([v for _, _, v in core.folds(tr)], ignore_index=True)
    oofs = {
        "home_only": core.cv_predict(tr, core.home_only),
        "elo": core.cv_predict(tr, core.elo),
        "xgboost": core.cv_predict(tr, core.xgboost(params)),
    }
    assert np.array_equal(oofs["xgboost"]["margin"].to_numpy(), val["margin"].to_numpy())
    sigma = {k: core.pooled_sigma(v) for k, v in oofs.items()}
    models = {k: core.evaluate(v, sigma[k]) for k, v in oofs.items()}
    check = {  # recomputed vs the saved dev report
        k: {"sigma_diff": sigma[k] - dev["models"][k]["sigma"],
            "mae_diff": models[k]["pooled"]["mae"] - dev["models"][k]["pooled"]["mae"]}
        for k in oofs
    }

    x = oofs["xgboost"]
    resid = (x["margin"] - x["mu"]).to_numpy()
    s = sigma["xgboost"]
    edges = np.arange(-50, 52, 2)
    counts, _ = np.histogram(np.clip(resid, -49.999, 49.999), bins=edges)
    expected = len(resid) * np.diff(norm.cdf(edges / s))
    hist = [{"lo": int(edges[i]), "hi": int(edges[i + 1]), "count": int(counts[i]), "expected": float(expected[i])}
            for i in range(len(counts))]

    def sd_groups(groups):
        return [{"label": k, "n": int(m.sum()), "sd": float(np.sqrt(np.mean(resid[m] ** 2)))} for k, m in groups if m.sum() >= 30]

    mu = x["mu"].to_numpy()
    qs = pd.qcut(mu, 5, labels=False)
    by_mu = sd_groups([(f"mu {mu[qs == i].min():.1f} to {mu[qs == i].max():.1f}", qs == i) for i in range(5)])
    st = val["seasonType"].to_numpy()
    by_type = sd_groups([(t, st == t) for t in ("Regular Season", "Playoffs", "Play-In")])
    by_season = sd_groups([(s_, x["season"].to_numpy() == s_) for s_ in core.VAL_SEASONS])

    def mu_vs_margin(oof):
        q = pd.qcut(oof["mu"], 12, labels=False, duplicates="drop")
        rows = []
        for _, t in oof.groupby(q):
            rows.append({"mu": float(t["mu"].mean()), "margin": float(t["margin"].mean()),
                         "se": float(t["margin"].std(ddof=1) / math.sqrt(len(t))), "n": len(t)})
        return rows

    err = {k: np.abs(v["margin"] - v["mu"]).to_numpy() for k, v in oofs.items()}
    brier = {k: (core.probabilities(v["mu"], sigma[k])["p_home_win"] - (v["margin"] > 0)) ** 2 for k, v in oofs.items()}
    var_margin = float(np.var(x["margin"]))
    return {
        "n_oof": len(x), "sigma": s, "params": params,
        "models": models,
        "calibration": {k: calib_tables(oofs[k]["margin"], oofs[k]["mu"], sigma[k]) for k in ("xgboost", "elo")},
        "mu_vs_margin": {k: mu_vs_margin(oofs[k]) for k in ("xgboost", "elo")},
        "sd_mu": float(np.std(mu)), "sd_margin": float(np.std(x["margin"])), "r2": 1 - float(np.mean(resid**2)) / var_margin,
        "residuals": {
            "hist": hist, "mean": float(resid.mean()), "skew": float(skew(resid)), "excess_kurtosis": float(kurtosis(resid)),
            "share_within_1sd": float(np.mean(np.abs(resid) <= s)), "share_within_2sd": float(np.mean(np.abs(resid) <= 2 * s)),
            "sd_by_mu": by_mu, "sd_by_type": by_type, "sd_by_season": by_season,
        },
        "boot_xgb_vs_elo": {"mae": boot_diff(err["xgboost"], err["elo"]), "brier_p_home_win": boot_diff(brier["xgboost"], brier["elo"])},
        "grid": dev["grid"],
        "ablation": ablation(tr, params, x),
        "recompute_check": check,
    }


def test_section(feat: pd.DataFrame, tr: pd.DataFrame, dev: dict, replay_summary: dict) -> tuple[dict, dict]:
    pred = pd.read_parquet(DATA / "replay" / "predictions.parquet")
    rows = feat.set_index("gameId").loc[pred["gameId"]]
    frozen, sigma, _ = load_champion(DATA)
    mu = {
        "chain": pred["mu"].to_numpy(),
        "frozen": core.predict_mu(frozen, rows),
        "elo": core.elo(tr, rows),
        "home_only": core.home_only(tr, rows),
    }
    sig = {"chain": sigma, "frozen": sigma, "elo": dev["models"]["elo"]["sigma"], "home_only": dev["models"]["home_only"]["sigma"]}
    y = pred["margin"].to_numpy()
    models = {k: core.metrics(y, mu[k], sig[k]) for k in mu}
    err = {k: np.abs(y - mu[k]) for k in mu}

    dates = pred["gameDate"].to_numpy()
    days = np.unique(dates)
    daily = []
    for d in days:
        win = (dates > d - np.timedelta64(30, "D")) & (dates <= d)
        cum = dates <= d
        row = {"date": str(pd.Timestamp(d).date()), "n_window": int(win.sum()), "n_cum": int(cum.sum())}
        for k in ("chain", "frozen", "elo"):
            row[f"roll_{k}"] = float(err[k][win].mean())
            row[f"cum_{k}"] = float(err[k][cum].mean())
        daily.append(row)

    types = pred["seasonType"].to_numpy()
    by_type = {
        t: {"n": int((types == t).sum()), **{k: float(err[k][types == t].mean()) for k in ("chain", "frozen", "elo", "home_only")}}
        for t in sorted(set(types))
    }
    first = pd.Timestamp(days[0])
    brier = {k: (core.probabilities(mu[k], sig[k])["p_home_win"] - (y > 0)) ** 2 for k in ("chain", "frozen", "elo")}

    # Checks against the replay report written by the pipeline.
    check = {
        "chain_mae_diff": models["chain"]["mae"] - replay_summary["champion_chain"]["mae"],
        "frozen_mae_diff": models["frozen"]["mae"] - replay_summary["frozen_v1"]["mae"],
        "elo_mae_diff": models["elo"]["mae"] - replay_summary["elo_baseline"]["mae"],
    }
    daily_saved = pd.read_parquet(DATA / "replay" / "daily.parquet")
    check["rolling_chain_max_diff"] = float(
        np.max(np.abs(daily_saved["rolling_mae_30d"].to_numpy() - np.array([r["roll_chain"] for r in daily])))
    )

    teams = pd.read_parquet(DATA / "raw" / max(p.name for p in (DATA / "raw").glob("snapshot=*")) / "teams")
    teams = teams.sort_values(["franchiseId", "endYear"]).groupby("franchiseId").tail(1)
    used = set(pred["homeFranchiseId"]) | set(pred["awayFranchiseId"])
    names = {r.franchiseId: r.teamName for r in teams.itertuples() if r.franchiseId in used}
    games = [
        {"d": str(pd.Timestamp(d).date()), "h": names[h], "a": names[a], "t": t, "y": int(m), "mu": a1, "muF": a2, "muE": a3,
         "pw": pw, "pb": pb, "pt": pt, "v": str(v)}
        for d, h, a, t, m, a1, a2, a3, pw, pb, pt, v in zip(
            pred["gameDate"], pred["homeFranchiseId"], pred["awayFranchiseId"], types, y, mu["chain"], mu["frozen"], mu["elo"],
            pred["p_home_win"], pred["p_blowout"], pred["p_tight"], pred["model_version"])
    ]
    test = {
        "n_games": len(y), "n_game_days": len(days), "first_date": str(first.date()), "last_date": str(pd.Timestamp(days[-1]).date()),
        "sigma": sigma, "sigma_elo": sig["elo"], "sd_mu": float(np.std(mu["chain"])), "sd_margin": float(np.std(y)),
        "models": models,
        "dev_mae": dev["models"]["xgboost"]["pooled"]["mae"],
        "boot_chain_vs_elo": {"mae": boot_diff(err["chain"], err["elo"]), "brier_p_home_win": boot_diff(brier["chain"], brier["elo"])},
        "boot_chain_vs_frozen": {"mae": boot_diff(err["chain"], err["frozen"]), "brier_p_home_win": boot_diff(brier["chain"], brier["frozen"])},
        "boot_frozen_vs_elo": {"mae": boot_diff(err["frozen"], err["elo"])},
        "calibration": {"chain": calib_tables(y, mu["chain"], sigma), "elo": calib_tables(y, mu["elo"], sig["elo"])},
        "mae_by_type": by_type,
        "residual_sd_by_type": {t: float(np.sqrt(np.mean((y[types == t] - mu["chain"][types == t]) ** 2))) for t in sorted(set(types))},
        "versions_used": {str(k): int(v) for k, v in pred["model_version"].value_counts().sort_index().items()},
        "recompute_check": check,
        "daily": daily,
    }
    return test, {"sigma": sigma, "games": games}


def checks_drift() -> tuple[dict, dict]:
    checks = pd.read_parquet(DATA / "replay" / "checks.parquet")
    drift = pd.read_parquet(DATA / "replay" / "drift.parquet")
    recs = []
    for r in checks.to_dict("records"):
        recs.append({k: (str(v.date()) if isinstance(v, pd.Timestamp) else v) for k, v in r.items()})
    if "flagged" not in drift:
        drift["flagged"] = drift["psi"] > 0.2
    d = drift[["check_date", "feature", "psi", "flagged"]].assign(check_date=lambda t: t["check_date"].astype(str).str[:10])
    checks_out = {"columns": list(checks.columns), "has_p_better": any(c.startswith("p_better") for c in checks.columns), "rows": recs}
    drift_out = {
        "dates": sorted(d["check_date"].unique()),
        "features": list(dict.fromkeys(d["feature"])),
        "columns": list(drift.columns),
        "rows": [[r.feature, r.check_date, r.psi, bool(r.flagged)] for r in d.itertuples()],
    }
    return checks_out, drift_out


def main() -> None:
    feat = pd.read_parquet(DATA / "features" / "features.parquet")
    tr = core.training_rows(feat)
    dev = json.loads((DATA / "reports" / "dev_cv.json").read_text())
    replay_summary = json.loads((DATA / "reports" / "replay.json").read_text())
    assert dev["feature_cols"] == core.FEATURE_COLS

    snap = max(p.name for p in (DATA / "raw").glob("snapshot=*"))
    size = lambda d: sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    sizes = {"raw_snapshot": size(DATA / "raw" / snap), "staged": size(DATA / "staged"), "features": size(DATA / "features")}
    write("timeline.json", timeline(feat, tr))
    write("features.json", feature_section(tr, dev))
    dev_out = dev_section(tr, dev)
    write("dev.json", dev_out)
    test, games = test_section(feat, tr, dev, replay_summary)
    write("test.json", test)
    write("games.json", games, nd=4)
    checks_out, drift_out = checks_drift()
    write("checks.json", checks_out)
    write("drift.json", drift_out, nd=4)
    write("meta.json", {
        "snapshot": snap.split("=", 1)[1], "bytes": sizes, "n_games_total": len(feat), "n_columns": feat.shape[1],
        "first_season": feat["season"].min(), "last_season": feat["season"].max(),
        "n_feature_cols": len(core.FEATURE_COLS), "n_train_rows": len(tr), "params": dev["selected_params"],
        "replay_check_columns": checks_out["columns"], "drift_columns": drift_out["columns"],
        "has_p_better": checks_out["has_p_better"],
        "recompute_check": {"dev": dev_out["recompute_check"], "test": test["recompute_check"]},
    })
    for p in sorted(OUT.glob("*.json")):
        print(f"{p.name:14s} {p.stat().st_size:>8,d} bytes")
    print("recompute check (should be ~0):", json.dumps(clean({"dev": dev_out["recompute_check"], "test": test["recompute_check"]}, 6)))


if __name__ == "__main__":
    main()
