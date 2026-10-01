#!/usr/bin/env python3
"""Recompute 96 features for the 995 training compositions.

Input: data/processed/training/consolidated_ground_state.csv.
Output: data/processed/training/consolidated_ground_state_features_v2.csv.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from herald.features.calculator import FeatureCalculator, load_element_properties

INPUT_FILE = PROJECT_ROOT / "data/processed/training/consolidated_ground_state.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/training/consolidated_ground_state_features_v2.csv"


def main():
    print("=" * 70)
    print("Recompute features for 995 training compositions (70D -> 96D)")
    print("=" * 70)

    # 1. Read the training set.
    print("\n[1/3] Reading the training set...")
    if not INPUT_FILE.exists():
        print(f"  File not found: {INPUT_FILE}")
        return

    train_data = pd.read_csv(INPUT_FILE)
    print(f"  Read {len(train_data)} samples")
    print(f"  Columns: {list(train_data.columns)[:10]}")

    # Check required columns.
    required_cols = ['M1', 'M2', 'M3', 'M4', 'delta_G_H']
    missing = [c for c in required_cols if c not in train_data.columns]
    if missing:
        print(f"  Missing required columns: {missing}")
        return

    # 2. Compute the new features.
    print("\n[2/3] Computing 96-dimensional features...")
    props = load_element_properties(PROJECT_ROOT / "data/elements/element_properties.json")
    calc = FeatureCalculator(props)

    features = calc.compute(train_data[['M1', 'M2', 'M3', 'M4']])
    print(f"  Feature matrix: {features.shape}")

    # 3. Merge and save.
    print("\n[3/3] Saving to CSV...")
    # Preserve metadata columns from the source table.
    meta_cols = ['M1', 'M2', 'M3', 'M4', 'delta_G_H']
    if 'source' in train_data.columns:
        meta_cols.append('source')
    if 'structure_id' in train_data.columns:
        meta_cols.append('structure_id')

    output_df = pd.concat([
        train_data[meta_cols],
        features
    ], axis=1)

    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE}")
    print(f"  Shape: {output_df.shape}")

    # Validate feature columns shared with the previous table.
    print("\n[Validation] Comparing the 13 M1 features...")
    old_m1_cols = [f"M1_{p}" for p in [
        "atomic_number", "atomic_weight", "atomic_radius", "electronegativity",
        "d_electrons", "valence_electrons", "d_band_center", "d_band_width",
        "work_function", "surface_energy", "lattice_constant", "melting_point",
        "cohesive_energy"
    ]]

    # Check whether the previous table contains these columns.
    old_feat_available = all(c in train_data.columns for c in old_m1_cols)

    if old_feat_available:
        errors = []
        for col in old_m1_cols:
            old_val = train_data[col].values
            new_val = output_df[col].values
            err = abs(old_val - new_val).max()
            errors.append(err)
            if err > 1e-3:
                print(f"  ✗ {col}: maximum error {err:.2e}")

        max_err = max(errors)
        print(f"  Maximum error: {max_err:.2e}")
        if max_err < 1e-3:
            print(f"  Shared-feature check passed (error <1e-3)")
        else:
            print(f"  Some feature errors exceed the threshold; check element-property versions")
    else:
        print(f"  Previous table has no M1 features; comparison skipped")

    print("\n" + "=" * 70)
    print("Training-feature calculation complete")
    print(f"  Input: {len(train_data)} samples")
    print(f"  Output: {output_df.shape}")
    print(f"  File: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
