"""Adaptive acquisition with an empirical chemical prior.

The prior is calibrated from labeled compositions and combines a
d-electron-count hit-rate term with a Gaussian electronegativity term.

    score(x) = alpha*U(x) + beta*E(x) + gamma*P_phys(x)
    U: normalized total predictive uncertainty.
    E: normalized proximity to delta_G_H = 0.
    P_phys: w1*pi_nd + w2*pi_chi.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PhysicsPrior:
    """Empirical chemical prior calibrated from labeled data."""

    h_by_nd: dict[int, float] = field(default_factory=dict)  # d-electron count -> hit rate
    chi_star: float = 0.0        # Mean M1 electronegativity among qualifying compositions
    tau_chi: float = 1.0         # Standard deviation of M1 electronegativity among qualifying compositions
    w_nd: float = 0.6            # Weight of the d-electron-count prior
    w_chi: float = 0.4           # Weight of the electronegativity prior
    excellent_thr: float = 0.2   # Activity threshold

    @classmethod
    def calibrate(
        cls,
        labeled_df: pd.DataFrame,
        props: dict,
        excellent_thr: float = 0.2,
        w_nd: float = 0.6,
        w_chi: float = 0.4,
    ) -> "PhysicsPrior":
        """Calibrate prior parameters from labeled compositions.

        Args:
            labeled_df: DataFrame containing M1 and delta_G_H.
            props: Element-property dictionary.
            excellent_thr: Qualifying threshold for |delta_G_H|.

        Returns:
            Calibrated PhysicsPrior.
        """
        df = labeled_df.copy()
        df["nd"] = df["M1"].map(lambda m: int(props[m]["d_electrons"]))
        df["chi"] = df["M1"].map(lambda m: float(props[m]["electronegativity"]))
        df["is_exc"] = df["delta_G_H"].abs() <= excellent_thr

        # Hit rates grouped by d-electron count
        h_by_nd = {}
        for nd, g in df.groupby("nd"):
            h_by_nd[int(nd)] = float(g["is_exc"].mean())

        # Electronegativity statistics among qualifying compositions
        exc = df[df["is_exc"]]
        chi_star = float(exc["chi"].mean()) if len(exc) > 0 else float(df["chi"].mean())
        tau_chi = float(exc["chi"].std()) if len(exc) > 1 else 1.0
        tau_chi = max(tau_chi, 0.1)  # Prevent an excessively narrow Gaussian.

        return cls(h_by_nd=h_by_nd, chi_star=chi_star, tau_chi=tau_chi,
                   w_nd=w_nd, w_chi=w_chi, excellent_thr=excellent_thr)

    def prior_nd(self, M1_list: np.ndarray, props: dict) -> np.ndarray:
        """Return the empirical d-electron-count prior pi_nd, normalized to [0,1]."""
        h_max = max(self.h_by_nd.values()) if self.h_by_nd else 1.0
        h_max = max(h_max, 1e-8)
        nd_arr = np.array([int(props[m]["d_electrons"]) for m in M1_list])
        return np.array([self.h_by_nd.get(int(nd), 0.0) / h_max for nd in nd_arr])

    def prior_chi(self, M1_list: np.ndarray, props: dict) -> np.ndarray:
        """Return the Gaussian electronegativity prior pi_chi."""
        chi = np.array([float(props[m]["electronegativity"]) for m in M1_list])
        return np.exp(-(chi - self.chi_star) ** 2 / (2 * self.tau_chi ** 2))

    def evaluate(self, M1_list: np.ndarray, props: dict) -> np.ndarray:
        """Return P_phys = w1*pi_nd + w2*pi_chi, normalized to [0,1]."""
        p_nd = self.prior_nd(M1_list, props)
        p_chi = self.prior_chi(M1_list, props)
        return self.w_nd * p_nd + self.w_chi * p_chi


def physics_guided_score(
    pred_mean: np.ndarray,
    uncertainty: np.ndarray,
    M1_list: np.ndarray,
    prior: PhysicsPrior,
    props: dict,
    alpha: float,
    gamma: float,
    target: float = 0.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute physics-guided acquisition scores.

    score = alpha*U + beta*E + gamma*P_phys, with beta = 1-alpha.

    Args:
        pred_mean: Predicted delta_G_H, shape (N,).
        uncertainty: Total predictive uncertainty, shape (N,).
        M1_list: Adsorption-center element symbols, shape (N,).
        prior: Calibrated PhysicsPrior.
        props: Element properties.
        alpha: Exploration weight in [0,1].
        gamma: Nonnegative chemical-prior weight.
        target: Target delta_G_H, default 0.

    Returns:
        Scores of shape (N,); higher scores take priority.
    """
    beta = 1.0 - alpha

    # Exploration term: normalized uncertainty
    U = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + eps)

    # Optimization term: proximity to the target
    dist = np.abs(pred_mean - target)
    E = 1.0 - (dist - dist.min()) / (dist.max() - dist.min() + eps)

    # Chemical-prior term
    P = prior.evaluate(M1_list, props)  # Already in [0,1]

    return alpha * U + beta * E + gamma * P


def compute_gamma(iteration: int, gamma_0: float = 0.3, tau_decay: float = 3.0) -> float:
    """Decay the prior weight exponentially: gamma(t) = gamma_0*exp(-t/tau).

    The prior receives greater weight in early rounds and less in later rounds.
    """
    return gamma_0 * np.exp(-iteration / tau_decay)


def greedy_diverse_selection(
    scores: np.ndarray,
    features: np.ndarray,
    n_select: int,
    diversity_weight: float = 0.5,
    candidate_pool: int = 500,
) -> np.ndarray:
    """Select a batch using scores and feature-space diversity.

    Greedily select n_select samples from the highest-scoring candidate_pool
    entries, balancing acquisition score and distance from selected samples.

    Args:
        scores: Acquisition scores, shape (N,).
        features: Features for distance calculation, shape (N,D).
        n_select: Number of samples to select.
        diversity_weight: Weight in [0,1]; larger values favor diversity.
        candidate_pool: Number of top-scoring candidates to consider.

    Returns:
        Original indices of the selected samples, shape (n_select,).
    """
    from scipy.spatial.distance import cdist

    # Restrict the pool to the highest-scoring candidates.
    pool_idx = np.argsort(scores)[::-1][:candidate_pool]
    pool_feat = features[pool_idx]
    pool_scores = scores[pool_idx]

    # Standardize features before computing distances.
    feat_mean = pool_feat.mean(axis=0)
    feat_std = pool_feat.std(axis=0) + 1e-8
    pool_feat_n = (pool_feat - feat_mean) / feat_std

    # Normalize scores.
    s_norm = (pool_scores - pool_scores.min()) / (pool_scores.max() - pool_scores.min() + 1e-10)

    selected = [0]  # Start with the highest-scoring candidate.
    for _ in range(n_select - 1):
        # Measure each candidate's minimum distance from the selected set.
        dists = cdist(pool_feat_n, pool_feat_n[selected])  # (pool, |selected|)
        min_dist = dists.min(axis=1)  # Distance to the nearest selected candidate
        min_dist_n = min_dist / (min_dist.max() + 1e-10)

        # Combined score = (1-w)*acquisition_score + w*diversity.
        combined = (1 - diversity_weight) * s_norm + diversity_weight * min_dist_n
        combined[selected] = -np.inf  # Exclude selected candidates.

        selected.append(int(np.argmax(combined)))

    return pool_idx[selected]
