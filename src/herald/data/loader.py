"""Load the composition space and training sets.

Sources:
    data/space/complete/phase_space_complete.h5:
        features/X: (57120, 77) feature matrix.
        features/feature_names: (77,) feature names.
        structures/m1..m4: (57120,) element symbols.
        structures/structure_str: (57120,) M1-M2-M3-M4 strings.
    data/processed/training/training_set_iterN.csv:
        M1-M4, delta_G_H and 77 feature columns.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_phase_space(h5_path: str | Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load the complete composition space.

    Returns:
        structures_df: 57120 rows with M1,M2,M3,M4,structure_str.
        X: Feature matrix of shape (57120,77).
        feature_names: List of 77 feature names.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        X = f["features/X"][:]
        feature_names = [
            n.decode() if isinstance(n, bytes) else n
            for n in f["features/feature_names"][:]
        ]
        m1 = [s.decode() for s in f["structures/m1"][:]]
        m2 = [s.decode() for s in f["structures/m2"][:]]
        m3 = [s.decode() for s in f["structures/m3"][:]]
        m4 = [s.decode() for s in f["structures/m4"][:]]
        struct_str = [s.decode() for s in f["structures/structure_str"][:]]

    structures = pd.DataFrame({
        "index": np.arange(len(m1)),
        "M1": m1, "M2": m2, "M3": m3, "M4": m4,
        "structure_str": struct_str,
    })
    return structures, X, feature_names


def load_training_set(
    csv_path: str | Path,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load a training set and separate features from labels.

    Args:
        csv_path: Path to training_set_iterN.csv.
        feature_names: The 77 names defining feature columns and order.

    Returns:
        X: Feature matrix of shape (n,77).
        y: delta_G_H labels of shape (n,).
        df: Complete source DataFrame.
    """
    df = pd.read_csv(csv_path)
    X = df[feature_names].values.astype(np.float64)
    y = df["delta_G_H"].values.astype(np.float64)
    return X, y, df
