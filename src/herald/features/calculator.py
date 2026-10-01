"""Composition features with separate M2 and symmetric M3/M4 inputs.

The multimetallic single-atom platform contains four M-N4 coordination
sites on N-doped graphene. M1 is the adsorption center, M2 is its neighbor,
and M3/M4 form the remote pair. Remote exchange is encoded through
means and absolute differences.

The 96D feature table contains 13 M1 properties, 13 M2 properties, six
remote-pair features, 52 global statistics and 12 combination features.
The model input selects 87 features from this table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Thirteen elemental properties
PROP13 = [
    "atomic_number", "atomic_weight", "atomic_radius", "electronegativity",
    "d_electrons", "valence_electrons", "d_band_center", "d_band_width",
    "work_function", "surface_energy", "lattice_constant", "melting_point",
    "cohesive_energy",
]


def load_element_properties(json_path: str | Path) -> dict:
    """Load element properties as {element_symbol: {property: value}}."""
    with open(json_path) as f:
        data = json.load(f)
    return data["elements"] if "elements" in data else data


class FeatureCalculator:
    """Construct 96D features with separate M2 and symmetric M3/M4 inputs.

    Blocks: 13 M1 properties, 13 M2 properties, six remote mean/absolute
    difference features, 52 global statistics and 12 combination features.
    The combination block includes five explicit formulas and seven
    retained placeholder columns.
    """

    def __init__(self, element_properties: dict):
        self.props = element_properties

    def _get_prop(self, element: str, prop: str) -> float:
        """Return a property value for one element."""
        return float(self.props[element][prop])

    def _prop_matrix(self, elements: list[str]) -> dict[str, np.ndarray]:
        """Return {property: np.array([M1,M2,M3,M4])}."""
        return {
            p: np.array([self._get_prop(e, p) for e in elements])
            for p in PROP13
        }

    # M1 site features (13D)

    def _m1_features(self, v: dict[str, np.ndarray]) -> dict[str, float]:
        """Return the 13 M1 elemental features."""
        return {f"M1_{p}": v[p][0] for p in PROP13}

    # M2 site features (13D)

    def _m2_features(self, v: dict[str, np.ndarray]) -> dict[str, float]:
        """Return the 13 M2 elemental properties.

        Separate M2 features preserve the neighboring-site identity.
        Related M1-M2 property differences also enter the global statistics.
        """
        feats = {}

        # Thirteen M2 elemental properties
        for p in PROP13:
            feats[f"M2_{p}"] = v[p][1]

        # M1-M2 interactions are also represented in global statistics.
        # Retained for compatibility with _statistical_features.
        # The global statistics already contain this information.

        return feats

    # Symmetric M3/M4 features (6D)

    def _m34_symmetric_features(self, v: dict[str, np.ndarray]) -> dict[str, float]:
        """Return six remote-pair features invariant to M3/M4 exchange.

        Use the mean and absolute difference for electronegativity,
        d-electron count and atomic radius.
        """
        # Select three elemental properties.
        key_props = ["electronegativity", "d_electrons", "atomic_radius"]
        feats = {}

        for p in key_props:
            v3 = v[p][2]  # Property value at M3
            v4 = v[p][3]  # Property value at M4
            feats[f"{p}_M34_mean"] = (v3 + v4) / 2.0
            feats[f"{p}_M34_absdiff"] = abs(v3 - v4)

        return feats

    # Global statistical features (52D)

    def _statistical_features(self, v: dict[str, np.ndarray]) -> dict[str, float]:
        """Return four statistics for each of 13 elemental properties.

        weighted_mean = 0.5*X(M1) + (X(M2)+X(M3)+X(M4))/6
        std = population standard deviation (ddof=0)
        range = max - min
        m1_deviation = X(M1) - mean(X(M2),X(M3),X(M4))
        """
        out = {}
        for p in PROP13:
            arr = v[p]
            out[f"{p}_weighted_mean"] = 0.5 * arr[0] + arr[1:].sum() / 6.0
            out[f"{p}_std"] = arr.std()
            out[f"{p}_range"] = arr.max() - arr.min()
            out[f"{p}_m1_deviation"] = arr[0] - arr[1:].mean()
        return out

    # Combination features (12D)

    def _combination_features(self, v: dict[str, np.ndarray]) -> dict[str, float]:
        """Return five explicit combination features and seven zero placeholders.

        Explicit formulas:
            M1_H_bond_polarity = |chi(M1)-2.20|.
            M1_d_band_filling = d_electrons(M1)/10.
            electronegativity_to_d_electrons_ratio = chi(M1)/d_e(M1).
            work_function_to_d_band_ratio = work_function(M1)/|epsilon_d(M1)|.
            atomic_size_mismatch = std(r)/mean(r).

        Columns set to 0.0:
            electron_affinity_M1, M1_work_function_deviation, gcn_deviation,
            neighbor_H_competition, lattice_strain_M1, surface_dipole_moment
            and electronic_configuration_index.
        """
        feats = {}

        # Five explicit feature formulas
        feats["M1_H_bond_polarity"] = abs(v["electronegativity"][0] - 2.20)
        feats["M1_d_band_filling"] = v["d_electrons"][0] / 10.0
        feats["electronegativity_to_d_electrons_ratio"] = (
            v["electronegativity"][0] / v["d_electrons"][0]
        )
        feats["work_function_to_d_band_ratio"] = (
            v["work_function"][0] / abs(v["d_band_center"][0])
        )
        feats["atomic_size_mismatch"] = (
            v["atomic_radius"].std() / v["atomic_radius"].mean()
        )

        # Seven retained columns use 0.0 placeholders.
        for name in [
            "electron_affinity_M1",
            "M1_work_function_deviation",
            "gcn_deviation",
            "neighbor_H_competition",
            "lattice_strain_M1",
            "surface_dipole_moment",
            "electronic_configuration_index",
        ]:
            feats[name] = 0.0

        return feats

    # Public interface

    def compute_row(self, m1: str, m2: str, m3: str, m4: str) -> dict[str, float]:
        """Compute all 96 features for one composition.

        Args:
            m1, m2, m3, m4: Element symbols, e.g. Ag, Co, Cu and Fe.

        Returns:
            Mapping from feature names to numeric values.
            Exchanging M3 and M4 leaves the feature vector unchanged.
        """
        v = self._prop_matrix([m1, m2, m3, m4])
        feats = {}
        feats.update(self._m1_features(v))
        feats.update(self._m2_features(v))
        feats.update(self._m34_symmetric_features(v))
        feats.update(self._statistical_features(v))
        feats.update(self._combination_features(v))
        return feats

    def compute(self, structures: pd.DataFrame) -> pd.DataFrame:
        """Compute features for a batch of compositions.

        Args:
            structures: DataFrame with M1, M2, M3 and M4 columns.

        Returns:
            DataFrame containing one feature vector per composition.
        """
        rows = [
            self.compute_row(r.M1, r.M2, r.M3, r.M4)
            for r in structures.itertuples()
        ]
        return pd.DataFrame(rows)

    def get_feature_names(self) -> list[str]:
        """Return feature names in calculation order."""
        # Obtain feature names from an example composition.
        dummy = self.compute_row("Ag", "Co", "Cu", "Fe")
        return list(dummy.keys())

    def get_feature_dim(self) -> int:
        """Return the feature dimension."""
        return len(self.get_feature_names())

    # Interfaces for symmetry-aware models

    def _global_statistics_vector(self, v: dict[str, np.ndarray]) -> np.ndarray:
        """Return the 52D global vector as a NumPy array.

        The ordering is PROP13 times four statistics, matching
        _statistical_features. Weighted mean, standard deviation, range
        and M1 deviation are invariant to M3/M4 exchange.
        """
        stats = []
        for p in PROP13:
            arr = v[p]
            stats.append(0.5 * arr[0] + arr[1:].sum() / 6.0)   # weighted_mean
            stats.append(arr.std())                             # std (ddof=0)
            stats.append(arr.max() - arr.min())                 # range
            stats.append(arr[0] - arr[1:].mean())               # m1_deviation
        return np.array(stats, dtype=np.float64)                # (52,)

    def compute_site_raw(
        self, m1: str, m2: str, m3: str, m4: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return raw site properties and global statistics for the shared-encoder model.

        M1/M2 use independent encoders. M3/M4 share an encoder followed
        by symmetric MaxPool. Global statistics describe all four metals.

        Args:
            m1, m2, m3, m4: Element symbols.

        Returns:
            x_M1, x_M2, x_M3, x_M4: PROP13 arrays of shape (13,).
            g_global: Exchange-invariant global statistics of shape (52,).
        """
        v = self._prop_matrix([m1, m2, m3, m4])
        x_m1 = np.array([v[p][0] for p in PROP13], dtype=np.float64)
        x_m2 = np.array([v[p][1] for p in PROP13], dtype=np.float64)
        x_m3 = np.array([v[p][2] for p in PROP13], dtype=np.float64)
        x_m4 = np.array([v[p][3] for p in PROP13], dtype=np.float64)
        g_global = self._global_statistics_vector(v)
        return x_m1, x_m2, x_m3, x_m4, g_global

    def compute_site_raw_batch(
        self, structures: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Compute batched shared-encoder inputs.

        Args:
            structures: DataFrame with M1, M2, M3 and M4 columns.

        Returns:
            x_M1, x_M2, x_M3 and x_M4, each with shape (N,13),
            and g_global with shape (N,52).
        """
        x1, x2, x3, x4, g = [], [], [], [], []
        for r in structures.itertuples():
            a, b, c, d, gg = self.compute_site_raw(r.M1, r.M2, r.M3, r.M4)
            x1.append(a); x2.append(b); x3.append(c); x4.append(d); g.append(gg)
        return {
            "x_M1": np.array(x1), "x_M2": np.array(x2),
            "x_M3": np.array(x3), "x_M4": np.array(x4),
            "g_global": np.array(g),
        }

    def compute_symmetric_input_from_features(
        self, feature_df: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Extract the 87D model input from the 96D feature table.

        Blocks are M1 (16 features), M2 (13), symmetric M3/M4 means
        and absolute differences (6), and global statistics (52).

        Args:
            feature_df: DataFrame containing the 96D composition features.

        Returns:
            {'x_M1': (N,16), 'x_M2': (N,13),
             'x_M34_sym': (N,6), 'g_global': (N,52)}.
        """
        # Sixteen M1 features
        M1_COLS = [f'M1_{p}' for p in PROP13] + [
            'M1_H_bond_polarity', 'M1_d_band_filling', 'M1_work_function_deviation'
        ]

        # Thirteen M2 elemental properties
        M2_COLS = [f'M2_{p}' for p in PROP13]

        # Six symmetric remote-pair features
        M34_PROPS = ['electronegativity', 'd_electrons', 'atomic_radius']
        M34_COLS = [f'{p}_M34_{stat}' for p in M34_PROPS for stat in ['mean', 'absdiff']]

        # Fifty-two global features: 13 properties times four statistics
        GLOBAL_STATS = ['mean', 'std', 'range', 'electronegativity_diff',
                       'd_electrons_diff', 'atomic_radius_diff',
                       'atomic_weight_diff', 'work_function_diff',
                       'surface_energy_diff', 'd_band_center_diff',
                       'cohesive_energy_diff', 'melting_point_diff']
        # Select columns outside the M1, M2 and M34 blocks.
        meta = ['M1', 'M2', 'M3', 'M4', 'delta_G_H']
        all_feat = [c for c in feature_df.columns if c not in meta]
        used = M1_COLS + M2_COLS + M34_COLS
        GLOBAL_COLS = [c for c in all_feat if c not in used][:52]  # Take the first 52 columns.

        return {
            'x_M1': feature_df[M1_COLS].values,
            'x_M2': feature_df[M2_COLS].values,
            'x_M34_sym': feature_df[M34_COLS].values,
            'g_global': feature_df[GLOBAL_COLS].values,
        }
