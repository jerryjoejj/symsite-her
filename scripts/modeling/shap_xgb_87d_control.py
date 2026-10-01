#!/usr/bin/env python3
"""Compute XGBoost attribution on the common 87-dimensional input.

TreeSHAP uses the same 995 compositions and site-index ranges as the
neural-network comparison. The script also reports five-fold errors for
the 87D and 96D feature sets and compares the saved 96D attribution table.
Outputs are saved under data/analysis/modeling/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shap_common import (  # noqa: E402
    FEATURE_NAMES_87, TRAIN_CSV, aggregate_by_site, load_87d, top_features,
)

OUT = ROOT / "data/analysis/modeling"
SEED = 42

# Fixed hyperparameters for the attribution model
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=SEED,
    n_jobs=-1,
)

_META_96 = ["phys_id", "M1", "M2", "M3", "M4", "E_clean", "E_H",
            "n_replicas", "sources", "delta_G_H"]


def load_96d() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load 96D features, excluding identity, energy and label columns."""
    df = pd.read_csv(TRAIN_CSV)
    cols = [c for c in df.columns if c not in _META_96]
    return df[cols].values, df["delta_G_H"].values, cols


def cv_compare(X96: np.ndarray, X87: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Compare five-fold MAE and R2 for the 96D and 87D feature sets."""
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for tag, X in [("96D_full", X96), ("87D_symmetric", X87)]:
        mae = -cross_val_score(xgb.XGBRegressor(**XGB_PARAMS), X, y,
                               cv=cv, scoring="neg_mean_absolute_error")
        r2 = cross_val_score(xgb.XGBRegressor(**XGB_PARAMS), X, y,
                             cv=cv, scoring="r2")
        rows.append({
            "feature_set": tag,
            "n_features": X.shape[1],
            "cv_mae_mean": float(mae.mean()),
            "cv_mae_std": float(mae.std()),
            "cv_r2_mean": float(r2.mean()),
            "cv_r2_std": float(r2.std()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 70)
    print("XGBoost 87D comparison: common neural-model features and site-index ranges")
    print("=" * 70)

    X87, y, _ = load_87d()
    X96, y96, cols96 = load_96d()
    assert np.allclose(y, y96), "Labels differ between the two loading paths"
    print(f"\nData: {X87.shape[0]} samples | 96D={X96.shape[1]}  87D={X87.shape[1]}")

    # Record columns unique to the 96D input in the metadata.
    extra = sorted(set(cols96) - set(FEATURE_NAMES_87))
    print(f"Columns unique to 96D ({len(extra)}): {', '.join(extra)}")

    print("\n[1/3] Five-fold CV: 96D vs 87D...")
    cv = cv_compare(X96, X87, y)
    print(cv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    d_mae = (cv.loc[1, "cv_mae_mean"] - cv.loc[0, "cv_mae_mean"]) * 1000
    print(f"  Change in MAE, 87D relative to 96D: {d_mae:+.1f} meV")

    print("\n[2/3] Training 87D XGBoost and exact TreeSHAP without background samples...")
    model = xgb.XGBRegressor(**XGB_PARAMS).fit(X87, y)
    sv = shap.TreeExplainer(model).shap_values(X87)
    print(f"  SHAP shape: {sv.shape}")

    print("\n[3/3] Aggregating site attribution using the same index ranges as the neural models...")
    by_site = aggregate_by_site(sv)
    print(by_site.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    top = top_features(sv, top_k=50)
    print("\nTop 10 features:")
    for _, r in top.head(10).iterrows():
        print(f"  {r['rank']:>2}. {r['feature']:<38} {r['site']:<9} {r['mean_abs_shap']:.4f}")

    # Quantify the attribution difference relative to the 96D reference.
    old = pd.read_csv(OUT / "shap_attribution_by_site.csv")
    old_pct = {r["site"]: float(r["percentage"]) for _, r in old.iterrows()}
    new_pct = by_site.set_index("site")["percentage"].to_dict()
    print("\nFeature-set attribution difference (XGBoost 87D - XGBoost 96D):")
    for s in ["M1", "M2", "M34_sym", "global"]:
        o = old_pct.get(s, 0.0)
        print(f"  {s:<10} 96D {o:6.2f}%  →  87D {new_pct[s]:6.2f}%  "
              f"({new_pct[s] - o:+6.2f} pp)")

    f_site = OUT / "shap_xgb_87d_by_site.csv"
    f_top = OUT / "shap_xgb_87d_top_features.csv"
    f_cv = OUT / "shap_xgb_87d_cv_comparison.csv"
    by_site.to_csv(f_site, index=False)
    top.to_csv(f_top, index=False)
    cv.to_csv(f_cv, index=False)

    meta = {
        "purpose": "XGBoost on the same 87D symmetric feature set as the two "
                   "neural networks, so all three models share one feature "
                   "space and one site partition.",
        "evaluation_samples": int(X87.shape[0]),
        "explainer": "TreeExplainer (exact; no background samples needed)",
        "seed": SEED,
        "xgb_params": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
        "site_aggregation": "index ranges from _shap_common.SITE_SLICES "
                            "(M1 0:16, M2 16:29, M34_sym 29:35, global 35:87); "
                            "identical to the neural-network scripts and "
                            "independent of feature names",
        "site_percentage_87D": {k: round(float(v), 2) for k, v in new_pct.items()},
        "site_percentage_96D_previous": {k: round(v, 2) for k, v in old_pct.items()},
        "columns_only_in_96D": extra,
        "cv_comparison": cv.to_dict(orient="records"),
        "notes": [
            "The 96D model remains the production screening surrogate "
            "(28,560 predictions, M1_valence_electrons SHAP=0.52, "
            "five-descriptor R^2=0.87); this 87D run is an added control, "
            "not a replacement.",
            "Two of the nine 96D-only columns (electron_affinity_M1, "
            "lattice_strain_M1) nominally belong to M1 but fell into the "
            "global bin under name-based grouping, biasing the 96D M1 share "
            "downward.",
        ],
    }
    (OUT / "shap_xgb_87d_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"\nSaved: {f_site}")
    print(f"Saved: {f_top}")
    print(f"Saved: {f_cv}")
    print(f"Saved: {OUT / 'shap_xgb_87d_meta.json'}")
    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
