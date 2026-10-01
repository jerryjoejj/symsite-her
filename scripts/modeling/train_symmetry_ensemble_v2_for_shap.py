#!/usr/bin/env python3
"""Train the symmetry-aware V2 ensemble for SHAP analysis.

Use the 995-sample dataset with an 80/20 training/validation split and
save the ten-member ensemble for attribution.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from herald.features.calculator import FeatureCalculator, load_element_properties
from herald.models.symmetry_aware_ensemble_v2 import (
    SymmetryAwareEnsembleV2, SymmetryEnsembleConfigV2
)

DATA_DIR = ROOT / "data/processed/training"
MODEL_DIR = ROOT / "models/ensemble"
SEED = 42


def main():
    print("=" * 70)
    print("Train the symmetry-aware V2 ensemble for SHAP analysis")
    print("=" * 70)

    # 1. Load data.
    print("\n1. Loading data...")
    df = pd.read_csv(DATA_DIR / 'consolidated_ground_state_features_v2.csv')
    y = df['delta_G_H'].values
    print(f"   Samples: {len(y)}")

    # 2. Construct 87D symmetric features.
    print("\n2. Computing symmetric features...")
    props_path = ROOT / 'data/elements/element_properties.json'
    props = load_element_properties(props_path)
    calc = FeatureCalculator(props)
    sym_dict = calc.compute_symmetric_input_from_features(df)

    print(f"   x_M1: {sym_dict['x_M1'].shape}")
    print(f"   x_M2: {sym_dict['x_M2'].shape}")
    print(f"   x_M34_sym: {sym_dict['x_M34_sym'].shape}")
    print(f"   g_global: {sym_dict['g_global'].shape}")

    # 3. Train the model.
    print("\n3. Training the V2 symmetric ensemble...")
    config = SymmetryEnsembleConfigV2(
        m1_dim=16,
        m2_dim=13,
        m34_sym_dim=6,
        global_dim=52,
        site_embed=32,
        fusion_dims=(128, 64),
        dropout=0.1,
        n_members=10,
        epochs=200,
        early_stop_patience=20,
        device='cuda',
        seed=SEED
    )

    model = SymmetryAwareEnsembleV2(config)

    # Use an 80/20 training/validation split.
    np.random.seed(SEED)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split = int(0.8 * len(y))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_data = {k: v[train_idx] for k, v in sym_dict.items()}
    val_data = {k: v[val_idx] for k, v in sym_dict.items()}

    print(f"   Training set: {len(train_idx)} samples")
    print(f"   Validation set: {len(val_idx)} samples")
    print(f"   Training...")

    model.fit(
        train_data, y[train_idx],
        val_data, y[val_idx],
        verbose=True
    )

    # 4. Save the model.
    print("\n4. Saving the model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / "symmetry_ensemble_v2_for_shap.pth"
    model.save(save_path)
    print(f"   Model saved: {save_path}")

    # 5. Evaluate predictions.
    print("\n5. Evaluating predictions...")
    pred = model.predict(val_data)
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y[val_idx], pred.mean)
    r2 = r2_score(y[val_idx], pred.mean)
    print(f"   Validation MAE: {mae:.4f} eV")
    print(f"   Validation R2: {r2:.4f}")
    print(f"   Mean epistemic uncertainty: {np.sqrt(pred.epistemic).mean():.4f} eV")
    print(f"   Mean aleatoric uncertainty: {np.sqrt(pred.aleatoric).mean():.4f} eV")

    print("\n" + "=" * 70)
    print("Training complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
