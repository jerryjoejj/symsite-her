#!/usr/bin/env python3
"""Evaluate V2 uncertainty calibration on 110 held-out compositions.

Use the current 87D production ensemble to predict the held-out set,
then compare epistemic uncertainty with prediction error.

Outputs:
    data/analysis/modeling/uncertainty_calibration_v3.csv
    data/analysis/modeling/uncertainty_buckets_v3.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from herald.features.calculator import FeatureCalculator, load_element_properties
from herald.models.symmetry_aware_ensemble_v2 import SymmetryAwareEnsembleV2

RES = ROOT / "data/analysis/modeling"
MODEL = ROOT / "models/ensemble/symmetry_ensemble_v2_production_v3.pt"
I9 = ROOT / "data/processed/cohorts/round_9/pred_vs_dft.csv"
I10 = ROOT / "data/processed/cohorts/round_10/pred_vs_dft.csv"


def main() -> None:
    d9 = pd.read_csv(I9)
    d10 = pd.read_csv(I10)

    rows = []
    for _, r in d9.iterrows():
        parts = str(r["formula"]).split("-")
        if len(parts) != 4:
            continue
        rows.append({"M1": parts[0], "M2": parts[1], "M3": parts[2],
                     "M4": parts[3], "dft": float(r["dft"])})
    for _, r in d10.iterrows():
        rows.append({"M1": str(r["M1"]), "M2": str(r["M2"]),
                     "M3": str(r["M3"]), "M4": str(r["M4"]),
                     "dft": float(r["dft"])})
    blind = pd.DataFrame(rows)

    props = load_element_properties(ROOT / "data/elements/element_properties.json")
    calc = FeatureCalculator(props)
    struct = blind[["M1", "M2", "M3", "M4"]].copy()
    feat96 = calc.compute(struct)
    blind_in = calc.compute_symmetric_input_from_features(
        struct.join(feat96))

    ens = SymmetryAwareEnsembleV2.load(MODEL)
    pred = ens.predict(blind_in)
    df = blind.copy()
    df["pred"] = pred.mean
    df["epi"] = np.sqrt(np.maximum(pred.epistemic, 0.0))
    df["abs_err"] = np.abs(df["pred"] - df["dft"])

    rho_s, p_s = stats.spearmanr(df["epi"], df["abs_err"])
    rho_p, p_p = stats.pearsonr(df["epi"], df["abs_err"])
    med = df["epi"].median()
    lo = df[df["epi"] <= med]["abs_err"].mean()
    hi = df[df["epi"] > med]["abs_err"].mean()

    df["bucket"] = pd.qcut(df["epi"], 3, labels=["low", "mid", "high"])
    bstat = df.groupby("bucket", observed=True).agg(
        n=("abs_err", "size"),
        mean_epi=("epi", "mean"),
        mae=("abs_err", "mean")).reset_index()
    mono = int(bstat["mae"].is_monotonic_increasing)

    print("=" * 60)
    print("V2 uncertainty calibration (110 held-out compositions, 87D)")
    print("=" * 60)
    print(f"Held-out compositions: {len(df)} (iter9 {len(d9)} + iter10 {len(d10)})")
    print(f"Overall MAE: {df['abs_err'].mean():.4f} eV")
    print(f"Epistemic range: [{df['epi'].min():.4f}, {df['epi'].max():.4f}]")
    print(f"Spearman ρ = {rho_s:.3f} (p={p_s:.3f})")
    print(f"Pearson  r = {rho_p:.3f} (p={p_p:.3f})")
    print(bstat.to_string(index=False))
    print(f"Monotonically increasing: {'yes' if mono else 'no'}")
    print(f"Lower-uncertainty half MAE: {lo:.4f}")
    print(f"Higher-uncertainty half MAE: {hi:.4f}")
    print(f"Ratio: {hi/lo:.2f}×")

    result = pd.DataFrame({
        "metric": ["n", "mae", "spearman_rho", "spearman_p", "pearson_r",
                   "mae_lo_epi", "mae_hi_epi", "ratio", "monotonic"],
        "value": [len(df), df["abs_err"].mean(), rho_s, p_s, rho_p,
                  lo, hi, hi / lo, mono]})
    result.to_csv(RES / "uncertainty_calibration_v3.csv", index=False)
    bstat.to_csv(RES / "uncertainty_buckets_v3.csv", index=False)
    df.to_csv(RES / "blindtest_v2_predictions_v3.csv", index=False,
              float_format="%.6f")
    print(f"saved {RES / 'uncertainty_calibration_v3.csv'}")
    print(f"saved {RES / 'uncertainty_buckets_v3.csv'}")
    print(f"saved {RES / 'blindtest_v2_predictions_v3.csv'}")


if __name__ == "__main__":
    main()
