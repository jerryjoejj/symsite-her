"""Initial sampling with clustering, random selection and a chemical prior.

The configurable default mix is 60% cluster diversity, 30% random
selection and 10% physics-informed selection. Sampling uses composition
features and elemental properties, independently of free-energy labels.
The physics-informed component uses a Pt-like d-band/electronegativity prior.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_diversity_selection(
    X: np.ndarray,
    n_samples: int,
    exclude: set[int] | None = None,
    random_state: int = 42,
) -> list[int]:
    """Select the sample nearest each K-means centroid for diversity sampling."""
    exclude = exclude or set()
    available = np.array([i for i in range(len(X)) if i not in exclude])
    Xa = X[available]

    Xs = StandardScaler().fit_transform(Xa)
    k = min(n_samples, len(available))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)

    selected = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(Xs[members] - centroid, axis=1)
        selected.append(int(available[members[np.argmin(dists)]]))
    return selected


def random_exploration_selection(
    n_total: int,
    n_samples: int,
    exclude: set[int] | None = None,
    random_state: int = 42,
) -> list[int]:
    """Select samples uniformly at random."""
    exclude = exclude or set()
    rng = np.random.default_rng(random_state)
    available = np.array([i for i in range(n_total) if i not in exclude])
    k = min(n_samples, len(available))
    return sorted(rng.choice(available, size=k, replace=False).tolist())


def physics_informed_selection(
    X: np.ndarray,
    feature_names: list[str],
    n_samples: int,
    exclude: set[int] | None = None,
    d_band_optimal: float = -2.25,
    en_optimal: float = 2.25,
) -> list[int]:
    """Select samples using the chemical prior, with a low default mixture weight."""
    exclude = exclude or set()
    try:
        d_idx = feature_names.index("d_band_center_weighted_mean")
        en_idx = feature_names.index("electronegativity_weighted_mean")
    except ValueError:
        return random_exploration_selection(len(X), n_samples, exclude)

    d_score = 1.0 / (1.0 + np.abs(X[:, d_idx] - d_band_optimal))
    en_score = 1.0 / (1.0 + np.abs(X[:, en_idx] - en_optimal))
    score = d_score + en_score

    order = np.argsort(score)[::-1]
    selected = [int(i) for i in order if i not in exclude][:n_samples]
    return selected


def mixed_initial_selection(
    X: np.ndarray,
    feature_names: list[str],
    n_total: int = 500,
    frac_cluster: float = 0.60,
    frac_random: float = 0.30,
    frac_physics: float = 0.10,
    random_state: int = 42,
) -> tuple[list[int], list[str]]:
    """Select an initial set using the configured strategy mixture.

    Returns:
        Selected indices and strategy labels.
    """
    n_cluster = int(round(n_total * frac_cluster))
    n_random = int(round(n_total * frac_random))
    n_physics = n_total - n_cluster - n_random

    idx_cluster = cluster_diversity_selection(X, n_cluster, random_state=random_state)
    idx_physics = physics_informed_selection(
        X, feature_names, n_physics, exclude=set(idx_cluster)
    )
    idx_random = random_exploration_selection(
        len(X), n_random, exclude=set(idx_cluster) | set(idx_physics),
        random_state=random_state,
    )

    indices = idx_cluster + idx_physics + idx_random
    labels = (
        ["cluster"] * len(idx_cluster)
        + ["physics"] * len(idx_physics)
        + ["random"] * len(idx_random)
    )
    return indices, labels
