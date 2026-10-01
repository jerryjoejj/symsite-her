"""Definitions, formulas and physical meanings of the 77D feature set."""

# Sixteen features for the adsorption center M1
M1_FEATURES = [
    "M1_atomic_number", "M1_atomic_weight", "M1_atomic_radius",
    "M1_electronegativity", "M1_d_electrons", "M1_valence_electrons",
    "M1_d_band_center", "M1_d_band_width", "M1_work_function",
    "M1_surface_energy", "M1_cohesive_energy", "M1_lattice_constant",
    "M1_melting_point", "M1_gcn", "M1_neighbor_coordination",
    "electron_affinity_M1",
]

# Sixty-one statistical features describing M1-M4 together
STATISTICAL_FEATURES = {
    "mean": ["atomic_number", "atomic_weight", "d_electrons", "electronegativity", "d_band_center", ...],
    "std": ["atomic_number", "atomic_weight", ...],
    "range": ["atomic_number", ...],
    "m1_deviation": ["atomic_number", ...],  # M1 deviation from the mean
}
