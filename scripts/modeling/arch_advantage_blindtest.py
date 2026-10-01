#!/usr/bin/env python3
"""Compare symmetric and flat models on 110 held-out compositions.

The held-out set includes OOD compositions and uses DFT reference labels.
Both models use the same 87D symmetric input and NLL loss; their
architectures and parameter counts differ.

Output: data/analysis/modeling/arch_advantage_blindtest.csv.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from herald.features.calculator import FeatureCalculator, load_element_properties
from herald.models.symmetry_aware_ensemble_v2 import (
    SymmetryAwareEnsembleV2, SymmetryEnsembleConfigV2)
from herald.models.flat_mlp_ensemble import FlatMLPEnsemble, FlatMLPConfig

TRAIN = ROOT / "data/processed/training/consolidated_ground_state_features_v2.csv"
I9 = ROOT / "data/processed/cohorts/round_9/pred_vs_dft.csv"
I10 = ROOT / "data/processed/cohorts/round_10/pred_vs_dft.csv"
OUT = ROOT / "data/analysis/modeling"
META = ['M1', 'M2', 'M3', 'M4', 'delta_G_H']
SEED = 42


def build_blindtest(calc):
    """Build the 110 held-out compositions and their DFT labels."""
    df9 = pd.read_csv(I9)
    parts = df9['formula'].str.split('-', expand=True)
    b9 = pd.DataFrame({'M1': parts[0], 'M2': parts[1], 'M3': parts[2],
                       'M4': parts[3], 'dft': df9['dft']})
    df10 = pd.read_csv(I10)
    b10 = pd.DataFrame({'M1': df10['M1'], 'M2': df10['M2'], 'M3': df10['M3'],
                        'M4': df10['M4'], 'dft': df10['dft']})
    return pd.concat([b9, b10], ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Held-out comparison: symmetric vs flat models (110 compositions)")
    print("=" * 60)

    calc = FeatureCalculator(load_element_properties(
        ROOT / "data/elements/element_properties.json"))

    # Training data with 87D symmetric features
    feat = pd.read_csv(TRAIN)
    y_tr = feat['delta_G_H'].values
    train_sym = calc.compute_symmetric_input_from_features(feat)
    X_tr_flat = np.concatenate([train_sym['x_M1'], train_sym['x_M2'],
                                train_sym['x_M34_sym'], train_sym['g_global']],
                               axis=1)

    # Construct 96D held-out features, then extract the 87D input.
    bt = build_blindtest(calc)
    print(f"\nHeld-out set: {len(bt)} compositions")
    # Compute the complete 96D features with the feature calculator.
    feat_cols = [c for c in feat.columns if c not in META]
    rows = []
    for r in bt.itertuples():
        f96 = calc.compute(pd.DataFrame([{
            'M1': r.M1, 'M2': r.M2, 'M3': r.M3, 'M4': r.M4}]))
        rows.append(f96.iloc[0])
    bt_feat = pd.DataFrame(rows)[feat_cols].reset_index(drop=True)
    bt_feat['M1'] = bt['M1'].values
    bt_feat['M2'] = bt['M2'].values
    bt_feat['M3'] = bt['M3'].values
    bt_feat['M4'] = bt['M4'].values
    y_bt = bt['dft'].values

    bt_sym = calc.compute_symmetric_input_from_features(bt_feat)
    X_bt_flat = np.concatenate([bt_sym['x_M1'], bt_sym['x_M2'],
                                bt_sym['x_M34_sym'], bt_sym['g_global']], axis=1)

    # Symmetric architecture
    print("\n[1/2] Training the symmetric model (87D, NLL)...")
    cfg_s = SymmetryEnsembleConfigV2(n_members=10, epochs=200,
                                     early_stop_patience=30, device='cpu')
    sym = SymmetryAwareEnsembleV2(cfg_s)
    sym.fit(train_sym, y_tr, verbose=False)
    pred_sym = sym.predict(bt_sym).mean
    ae_sym = np.abs(pred_sym - y_bt)

    # Flat architecture
    print("[2/2] Training the flat MLP (87D, NLL)...")
    cfg_f = FlatMLPConfig(in_dim=X_tr_flat.shape[1], loss_type='nll',
                          n_members=10, epochs=200, dropout=0.1,
                          weight_decay=1e-5, device='cpu')
    flat = FlatMLPEnsemble(cfg_f)
    flat.fit(X_tr_flat, y_tr)
    pred_flat = flat.predict(X_bt_flat)
    ae_flat = np.abs(pred_flat - y_bt)

    # Comparison
    print("\n" + "=" * 60)
    print("Held-out results: symmetric vs flat (110 compositions, 87D, NLL)")
    print("=" * 60)
    print(f"Symmetric-model MAE: {ae_sym.mean():.4f}")
    print(f"Flat-MLP MAE: {ae_flat.mean():.4f}")
    print(f"Delta MAE (symmetric - flat): {ae_sym.mean()-ae_flat.mean():+.4f}")

    t, p = stats.ttest_rel(ae_sym, ae_flat)
    w, pw = stats.wilcoxon(ae_sym, ae_flat)
    print(f"\nPaired t test: t={t:.2f}, p={p:.3f}")
    print(f"Wilcoxon:   p={pw:.3f}")
    sig = "significant" if p < 0.05 else "not significant"
    print(f"Held-out architecture comparison [{sig}]")

    pd.DataFrame({
        'M1': bt['M1'], 'M2': bt['M2'], 'M3': bt['M3'], 'M4': bt['M4'],
        'dft': y_bt, 'pred_sym': pred_sym, 'pred_flat': pred_flat,
        'ae_sym': ae_sym, 'ae_flat': ae_flat,
    }).to_csv(OUT / "arch_advantage_blindtest.csv", index=False)
    print(f"\nSaved: {OUT}/arch_advantage_blindtest.csv")


if __name__ == "__main__":
    main()
