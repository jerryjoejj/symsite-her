"""Shared SHAP utilities for the 87-dimensional site-grouped input.

Feature names follow calculator.PROP13 and the training-table columns.
SITE_SLICES defines the M1/M2/M34/global partition used for attribution.
The module provides data loading, site aggregation and feature ranking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from herald.features.calculator import PROP13, load_element_properties  # noqa: E402

TRAIN_CSV = ROOT / "data/processed/training/consolidated_ground_state_features_v2.csv"
ELEMENT_PROPS = ROOT / "data/elements/element_properties.json"

# Use the same feature groups as compute_symmetric_input_from_features().
M1_COLS = [f"M1_{p}" for p in PROP13] + [
    "M1_H_bond_polarity", "M1_d_band_filling", "M1_work_function_deviation",
]
M2_COLS = [f"M2_{p}" for p in PROP13]
M34_PROPS = ["electronegativity", "d_electrons", "atomic_radius"]
M34_COLS = [f"{p}_M34_{s}" for p in M34_PROPS for s in ["mean", "absdiff"]]
_META = ["M1", "M2", "M3", "M4", "delta_G_H"]

SITE_SLICES: dict[str, slice] = {
    "M1": slice(0, 16),
    "M2": slice(16, 29),
    "M34_sym": slice(29, 35),
    "global": slice(35, 87),
}


def _global_cols(columns: list[str]) -> list[str]:
    """Match calculator.py GLOBAL_COLS: exclude metadata and used columns, then take the first 52."""
    used = set(M1_COLS + M2_COLS + M34_COLS)
    return [c for c in columns if c not in _META and c not in used][:52]


def feature_names_87(train_csv: Path = TRAIN_CSV) -> list[str]:
    """Return the 87D feature names in concatenation order."""
    cols = pd.read_csv(train_csv, nrows=0).columns.tolist()
    names = M1_COLS + M2_COLS + M34_COLS + _global_cols(cols)
    if len(names) != 87:
        raise ValueError(f"Feature-name count {len(names)} != 87; check the feature-table header")
    return names


FEATURE_NAMES_87: list[str] = feature_names_87()

SITE_LABELS: list[str] = [
    site for site, sl in SITE_SLICES.items() for _ in range(sl.stop - sl.start)
]


def load_87d(train_csv: Path = TRAIN_CSV) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the 87D symmetric features, labels and source DataFrame.

    Returns:
        X: (N, 87), ordered as M1|M2|M34_sym|global.
        y: (N,) delta_G_H.
        df: Source DataFrame, including M1..M4 element symbols.
    """
    from herald.features.calculator import FeatureCalculator

    df = pd.read_csv(train_csv)
    y = df["delta_G_H"].values
    calc = FeatureCalculator(load_element_properties(ELEMENT_PROPS))
    d = calc.compute_symmetric_input_from_features(df)
    X = np.hstack([d["x_M1"], d["x_M2"], d["x_M34_sym"], d["g_global"]])
    if X.shape[1] != 87:
        raise ValueError(f"Feature dimension {X.shape[1]} != 87")
    return X, y, df


def aggregate_by_site(shap_values: np.ndarray) -> pd.DataFrame:
    """Aggregate absolute SHAP values by site.

    Args:
        shap_values: Array of shape (N, 87).

    Returns:
        DataFrame[site, n_features, mean_abs_shap, percentage].
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    rows = []
    for site, sl in SITE_SLICES.items():
        rows.append({
            "site": site,
            "n_features": sl.stop - sl.start,
            "mean_abs_shap": float(mean_abs[sl].sum()),
        })
    out = pd.DataFrame(rows)
    out["percentage"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum() * 100
    return out.sort_values("percentage", ascending=False).reset_index(drop=True)


def top_features(shap_values: np.ndarray, top_k: int = 50) -> pd.DataFrame:
    """Return the top-k features ranked by absolute SHAP, with site labels."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": FEATURE_NAMES_87,
        "site": SITE_LABELS,
        "mean_abs_shap": mean_abs,
    }).nlargest(top_k, "mean_abs_shap").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df
