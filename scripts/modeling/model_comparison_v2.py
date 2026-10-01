#!/usr/bin/env python3
"""Model tuning and comparison on a common 87D input.

Use the same five-fold splits (seed 42) for the 995 training compositions,
with scaling fitted within each training fold. Compare XGBoost grid search,
small grids for the flat and symmetry-aware ensembles, and ten traditional
models using their selected hyperparameters.

Outputs:
    data/analysis/modeling/model_comparison_v2.csv
    data/analysis/modeling/model_comparison_v2.json
    data/analysis/modeling/tuned_models/ (fold predictions and settings)
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import BayesianRidge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from herald.features.calculator import FeatureCalculator, load_element_properties
from herald.models.symmetry_aware_ensemble_v2 import (
    SymmetryAwareEnsembleV2,
    SymmetryEnsembleConfigV2,
)
from herald.models.flat_mlp_ensemble import FlatMLPEnsemble, FlatMLPConfig

TRAIN = ROOT / "data/processed/training/consolidated_ground_state_features_v2.csv"
ELEM = ROOT / "data/elements/element_properties.json"
TRAD_BEST = ROOT / "data/traditional_machine_learning"
OUT = ROOT / "data/analysis/modeling/tuned_models"
N_FOLDS = 5
SEED = 42


def load_87d() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    feat = pd.read_csv(TRAIN)
    y = feat["delta_G_H"].values
    calc = FeatureCalculator(load_element_properties(ELEM))
    data = calc.compute_symmetric_input_from_features(feat)
    X87 = np.concatenate(
        [data["x_M1"], data["x_M2"], data["x_M34_sym"], data["g_global"]],
        axis=1)
    assert X87.shape[1] == 87, X87.shape
    return data, X87, y


def cv_mae_folds(pred_folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    maes = [mean_absolute_error(y, p) for y, p in pred_folds]
    return float(np.mean(maes)), float(np.std(maes, ddof=0))


def tune_xgboost(X87: np.ndarray, y: np.ndarray, kf: KFold) -> dict:
    grid = [
        dict(n_estimators=ne, max_depth=md, learning_rate=lr,
             subsample=ss, colsample_bytree=cs, min_child_weight=mcw,
             random_state=SEED, n_jobs=-1)
        for ne in (300, 500)
        for md in (4, 7)
        for lr in (0.04, 0.08)
        for ss in (0.8, 1.0)
        for cs in (0.8, 1.0)
        for mcw in (1, 5)
    ]
    best_cfg, best_score, best_folds = None, float("inf"), None
    for cfg in grid:
        folds = []
        for tr, va in kf.split(X87):
            sc = StandardScaler().fit(X87[tr])
            m = xgb.XGBRegressor(**cfg).fit(sc.transform(X87[tr]), y[tr])
            folds.append((y[va], m.predict(sc.transform(X87[va]))))
        mean_mae, _ = cv_mae_folds(folds)
        if mean_mae < best_score:
            best_score, best_cfg, best_folds = mean_mae, cfg, folds
    mae, sd = cv_mae_folds(best_folds)
    all_pred = np.concatenate([p for _, p in best_folds])
    all_y = np.concatenate([y_ for y_, _ in best_folds])
    r2 = r2_score(all_y, all_pred)
    return {"method": "XGBoost", "mae_mean": mae, "mae_std": sd,
            "r2": r2, "best_params": best_cfg, "folds": best_folds}


def tune_flat_mlp(X87: np.ndarray, y: np.ndarray, kf: KFold) -> dict:
    grid = [
        dict(hidden=h, lr=lr, weight_decay=wd)
        for h in ((128, 64), (256, 128, 64))
        for lr in (3e-4, 1e-3)
        for wd in (1e-5, 1e-4)
    ]
    best_cfg, best_score, best_folds = None, float("inf"), None
    for cfg in grid:
        folds = []
        for tr, va in kf.split(X87):
            m = FlatMLPEnsemble(FlatMLPConfig(
                hidden=cfg["hidden"], lr=cfg["lr"], dropout=0.1,
                weight_decay=cfg["weight_decay"], n_members=10, epochs=200,
                early_stop_patience=30, seed=SEED))
            m.fit(X87[tr], y[tr], X87[va], y[va])
            folds.append((y[va], m.predict(X87[va])))
        mean_mae, _ = cv_mae_folds(folds)
        if mean_mae < best_score:
            best_score, best_cfg, best_folds = mean_mae, cfg, folds
    mae, sd = cv_mae_folds(best_folds)
    all_pred = np.concatenate([p for _, p in best_folds])
    all_y = np.concatenate([y_ for y_, _ in best_folds])
    r2 = r2_score(all_y, all_pred)
    return {"method": "Flat MLP", "mae_mean": mae, "mae_std": sd,
            "r2": r2, "best_params": best_cfg, "folds": best_folds}


def tune_symmetry_v2(
    data: dict[str, np.ndarray], y: np.ndarray, kf: KFold
) -> dict:
    grid = [
        dict(fusion=f, site_embed=e, lr=lr)
        for f in ((64, 32), (128, 64))
        for e in (16, 32)
        for lr in (3e-4, 1e-3)
    ]
    best_cfg, best_score, best_folds = None, float("inf"), None
    for cfg in grid:
        folds = []
        for tr, va in kf.split(y):
            tr_d = {k: v[tr] for k, v in data.items()}
            va_d = {k: v[va] for k, v in data.items()}
            m = SymmetryAwareEnsembleV2(SymmetryEnsembleConfigV2(
                fusion_dims=cfg["fusion"], site_embed=cfg["site_embed"],
                lr=cfg["lr"], dropout=0.1, n_members=10, epochs=200,
                early_stop_patience=30, seed=SEED))
            m.fit(tr_d, y[tr], va_d, y[va], verbose=False)
            folds.append((y[va], m.predict(va_d).mean))
        mean_mae, _ = cv_mae_folds(folds)
        if mean_mae < best_score:
            best_score, best_cfg, best_folds = mean_mae, cfg, folds
    mae, sd = cv_mae_folds(best_folds)
    all_pred = np.concatenate([p for _, p in best_folds])
    all_y = np.concatenate([y_ for y_, _ in best_folds])
    r2 = r2_score(all_y, all_pred)
    return {"method": "Symmetry-aware ensemble V2", "mae_mean": mae,
            "mae_std": sd, "r2": r2, "best_params": best_cfg,
            "folds": best_folds}


def traditional_models() -> list[tuple[str, object]]:
    """Construct ten traditional models with the selected hyperparameters and native Python types."""
    gpr_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + \
        WhiteKernel(noise_level=1e-5)
    return [
        ("LightGBM", lgb.LGBMRegressor(
            learning_rate=0.01, max_depth=7, min_child_samples=5,
            n_estimators=1000, num_leaves=31, random_state=SEED,
            verbose=-1, n_jobs=1)),
        ("CatBoost", cb.CatBoostRegressor(
            depth=4, iterations=300, l2_leaf_reg=1, learning_rate=0.1,
            random_state=SEED, verbose=0, thread_count=1)),
        ("GradientBoosting", GradientBoostingRegressor(
            learning_rate=0.1, max_depth=5, min_samples_leaf=4,
            min_samples_split=10, n_estimators=300, random_state=SEED)),
        ("RandomForest", RandomForestRegressor(
            max_depth=10, max_features="sqrt", min_samples_leaf=1,
            min_samples_split=2, n_estimators=100, random_state=SEED,
            n_jobs=1)),
        ("SVR", SVR(C=100, epsilon=0.01, kernel="rbf")),
        ("GPR", GaussianProcessRegressor(
            kernel=gpr_kernel, alpha=1e-5, random_state=SEED)),
        ("DecisionTree", DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=4, min_samples_split=2,
            splitter="random", random_state=SEED)),
        ("AdaBoost", AdaBoostRegressor(
            learning_rate=0.1, loss="linear", n_estimators=50,
            random_state=SEED)),
        ("BayesianRidge", BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-4, lambda_1=1e-4, lambda_2=1e-6)),
        ("Lasso", Lasso(alpha=0.1, random_state=SEED, max_iter=5000)),
    ]


def eval_traditional(X87: np.ndarray, y: np.ndarray, kf: KFold) -> list[dict]:
    results = []
    for name, model in traditional_models():
        folds = []
        for tr, va in kf.split(X87):
            sc = StandardScaler().fit(X87[tr])
            m = clone(model)
            m.fit(sc.transform(X87[tr]), y[tr])
            folds.append((y[va], m.predict(sc.transform(X87[va]))))
        mae, sd = cv_mae_folds(folds)
        all_pred = np.concatenate([p for _, p in folds])
        all_y = np.concatenate([y_ for y_, _ in folds])
        r2 = r2_score(all_y, all_pred)
        results.append({"method": name, "mae_mean": mae, "mae_std": sd,
                        "r2": r2, "best_params": model.get_params(),
                        "folds": folds})
    return results


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, X87, y = load_87d()
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    print(f"data: {len(y)} × 87D, shared KFold(5, shuffle, seed=42)\n")

    print("tuning XGBoost ...", flush=True)
    t0 = time.time()
    xgb_res = tune_xgboost(X87, y, kf)
    print(f"  XGBoost MAE={xgb_res['mae_mean']:.4f} ± {xgb_res['mae_std']:.4f} "
          f"({time.time()-t0:.0f}s) params={xgb_res['best_params']}", flush=True)

    print("tuning Flat MLP ...", flush=True)
    t0 = time.time()
    flat_res = tune_flat_mlp(X87, y, kf)
    print(f"  Flat MLP MAE={flat_res['mae_mean']:.4f} ± {flat_res['mae_std']:.4f} "
          f"({time.time()-t0:.0f}s) params={flat_res['best_params']}", flush=True)

    print("tuning Symmetry V2 ...", flush=True)
    t0 = time.time()
    sym_res = tune_symmetry_v2(data, y, kf)
    print(f"  Sym V2 MAE={sym_res['mae_mean']:.4f} ± {sym_res['mae_std']:.4f} "
          f"({time.time()-t0:.0f}s) params={sym_res['best_params']}", flush=True)

    print("re-evaluating traditional ML ...", flush=True)
    trad_res = eval_traditional(X87, y, kf)
    for r in trad_res:
        print(f"  {r['method']:18s} MAE={r['mae_mean']:.4f} ± {r['mae_std']:.4f}",
              flush=True)

    all_res = [xgb_res, flat_res, sym_res] + trad_res
    df = pd.DataFrame([{
        "model": r["method"],
        "mae_mean": r["mae_mean"],
        "mae_std": r["mae_std"],
        "r2": r["r2"],
        "best_params": json.dumps(r["best_params"], default=str),
    } for r in all_res]).sort_values("mae_mean").reset_index(drop=True)
    df.to_csv(ROOT / "data/analysis/modeling/model_comparison_v2.csv", index=False,
              float_format="%.5f")

    # Save each model's fold predictions.
    for r in all_res:
        frows = []
        for i, (yv, pv) in enumerate(r["folds"]):
            for j, (yt, pt) in enumerate(zip(yv, pv)):
                frows.append({"fold": i, "y_true": yt, "y_pred": pt})
        pd.DataFrame(frows).to_csv(
            OUT / f"{r['method'].replace(' ', '_')}_cv.csv", index=False)

    summary = {
        "protocol": "87D symmetric input, KFold(5, shuffle=True, random_state=42), "
                    "fold-internal StandardScaler; neural models use validation-fold "
                    "early stopping; XGBoost/traditional ML tuned by grid search.",
        "n_samples": int(len(y)),
        "metric": "MAE (eV)",
        "ranking": [
            {"model": r["method"], "mae_mean": r["mae_mean"],
             "mae_std": r["mae_std"], "r2": r["r2"],
             "best_params": r["best_params"]}
            for r in sorted(all_res, key=lambda z: z["mae_mean"])
        ],
    }
    (ROOT / "data/analysis/modeling/model_comparison_v2.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8")
    print("\n" + "=" * 70)
    print("final ranking (MAE)")
    print(df[["model", "mae_mean", "mae_std", "r2"]].to_string(index=False))
    print(f"\nsaved model_comparison_v2.csv + json")


if __name__ == "__main__":
    main()
