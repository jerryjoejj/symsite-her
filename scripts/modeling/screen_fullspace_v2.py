#!/usr/bin/env python3
"""Train the 87D symmetry-aware V2 ensemble and predict the full space.

Fit ten members to all 995 training compositions and save pred_dG and
epistemic uncertainty in activity_map_28560_v3.csv.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from herald.features.calculator import FeatureCalculator, load_element_properties
from herald.models.symmetry_aware_ensemble_v2 import (
    SymmetryAwareEnsembleV2,
    SymmetryEnsembleConfigV2,
)

TRAIN = ROOT / "data/processed/training/consolidated_ground_state_features_v2.csv"
PHASE_H5 = ROOT / "data/space/features_28560_v2.h5"
PROPS = ROOT / "data/elements/element_properties.json"
OUT = ROOT / "data/analysis/modeling"
MODEL_OUT = ROOT / "models/ensemble/symmetry_ensemble_v2_production_v3.pt"
MODEL_META = ROOT / "models/ensemble/symmetry_ensemble_v2_production_v3_meta.json"
META = ["M1", "M2", "M3", "M4", "delta_G_H"]
SEED = 42


def main() -> None:
    feat = pd.read_csv(TRAIN)
    feat_cols = [c for c in feat.columns if c not in META]
    y = feat["delta_G_H"].values
    calc = FeatureCalculator(load_element_properties(PROPS))
    train_in = calc.compute_symmetric_input_from_features(feat)
    print(f"training set: {len(y)} samples, 87D symmetric input")

    cfg = SymmetryEnsembleConfigV2(
        n_members=10, epochs=300, batch_size=64, early_stop_patience=30,
        device="cuda" if torch.cuda.is_available() else "cpu", seed=SEED)
    ens = SymmetryAwareEnsembleV2(cfg)
    ens.fit(train_in, y, verbose=False)
    print("V2 ensemble trained")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    ens.save(MODEL_OUT)
    pred_train = ens.predict(train_in)
    train_mae = float(np.abs(pred_train.mean - y).mean())
    MODEL_META.write_text(json.dumps({
        "architecture": "SymmetryAwareEnsembleV2",
        "n_train": len(y),
        "n_members": cfg.n_members,
        "epochs": cfg.epochs,
        "site_embed": cfg.site_embed,
        "fusion_dims": list(cfg.fusion_dims),
        "dropout": cfg.dropout,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "input_dims": {
            "M1": train_in["x_M1"].shape[1],
            "M2": train_in["x_M2"].shape[1],
            "M34_sym": train_in["x_M34_sym"].shape[1],
            "global": train_in["g_global"].shape[1],
            "total": sum(v.shape[1] for v in train_in.values()),
        },
        "train_mae": train_mae,
    }, indent=2), encoding="utf-8")
    print(f"saved model: {MODEL_OUT} (train MAE {train_mae:.4f} eV)")

    with h5py.File(PHASE_H5, "r") as f:
        X_phase = f["features/matrix"][:]
        h5_names = [n.decode() if isinstance(n, bytes) else str(n)
                    for n in f["features/names"][:]]
        M = {k: [s.decode() if isinstance(s, bytes) else str(s)
                 for s in f[f"structures/{k}"][:]]
             for k in ["M1", "M2", "M3", "M4"]}
    assert h5_names == feat_cols, "feature order mismatch"

    phase_df = pd.DataFrame({**M})
    for i, c in enumerate(feat_cols):
        phase_df[c] = X_phase[:, i]
    phase_in = calc.compute_symmetric_input_from_features(phase_df)
    pred = ens.predict(phase_in)
    pred_dG = pred.mean
    epi = np.sqrt(np.maximum(pred.epistemic, 0.0))

    activity = pd.DataFrame({
        "M1": M["M1"], "M2": M["M2"], "M3": M["M3"], "M4": M["M4"],
        "pred_dG": pred_dG, "epistemic": epi, "abs_dG": np.abs(pred_dG),
    }).sort_values("abs_dG").reset_index(drop=True)
    activity.to_csv(OUT / "activity_map_28560_v3.csv", index=False)
    print(f"saved activity_map_28560_v3.csv ({len(activity)} rows)")

    activity.head(30).to_csv(OUT / "top_candidates_v3.csv", index=False)
    print("saved top_candidates_v3.csv")


if __name__ == "__main__":
    main()
