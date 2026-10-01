"""Merge composition records, 77D features and DFT free-energy labels.

Join compositions and DFT results on index and M1..M4, then concatenate
features in row order, matching the HDF5 feature ordering.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Metadata columns excluded from the feature matrix
META_COLUMNS = [
    "index", "M1", "M2", "M3", "M4", "strategy", "sample_id",
    "E_clean", "E_H", "delta_E_H", "delta_G_H", "status",
]


def merge_training_set(
    structures: pd.DataFrame,
    features: pd.DataFrame,
    dft_results: pd.DataFrame,
) -> pd.DataFrame:
    """Merge composition, feature and DFT tables into a training set.

    Args:
        structures: index, M1..M4 and optional strategy fields.
        features: 77D features in the same row order as structures.
        dft_results: index, M1..M4, delta_G_H and other result fields.

    Returns:
        DataFrame with metadata and 77 features per row.
    """
    merge_keys = ["index", "M1", "M2", "M3", "M4"]
    data = structures.merge(dft_results, on=merge_keys, how="inner")

    if len(data) != len(structures):
        raise ValueError(
            f"Composition/DFT merge changed the row count: "
            f"structures={len(structures)}, merged={len(data)}"
        )

    # Concatenate features in the same row order as structures.
    if len(features) != len(data):
        raise ValueError(
            f"Feature and merged-result row counts differ: features={len(features)}, merged={len(data)}"
        )

    feat = features.reset_index(drop=True)
    data = data.reset_index(drop=True)
    combined = pd.concat([data, feat], axis=1)
    return combined


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature-column names, excluding metadata."""
    return [c for c in df.columns if c not in META_COLUMNS]


def load_cumulative_training_set(
    iteration_dirs: list[Path],
    label_col: str = "delta_G_H",
) -> pd.DataFrame:
    """Build a cumulative training set from iteration merged_data.csv files.

    Args:
        iteration_dirs: Processed-result directories for the iterations.
        label_col: Label-column name.

    Returns:
        Concatenated training data, deduplicated by index.
    """
    frames = []
    for d in iteration_dirs:
        path = Path(d) / "merged_data.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No merged_data.csv files found")

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.drop_duplicates(subset=["index"], keep="first")
    combined = combined[combined[label_col].notna()].reset_index(drop=True)
    return combined
