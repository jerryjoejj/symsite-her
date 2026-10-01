"""Adaptive acquisition weights for exploration and optimization.

    score = alpha * uncertainty_norm + beta * (1 - |pred_delta_G_H|_norm)
    alpha + beta = 1.

The exploration weight combines an annealing schedule, a sample-budget
factor and an increase when improvement stalls. Early rounds favor
exploration; later rounds increasingly target delta_G_H near zero.

References:
    Settles, Active Learning Literature Survey, 2009.
    Kirkpatrick et al., Science 1983 (annealing).
    Auer et al., 2002 (UCB trade-off).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AcquisitionConfig:
    """Configuration for adaptive acquisition."""

    alpha_start: float = 0.7  # Initial exploration weight (iteration 1)
    alpha_min: float = 0.2  # Minimum exploration weight
    alpha_max: float = 0.9  # Maximum exploration weight
    decay: float = 0.74  # Per-round decay factor (0.7 -> about 0.52 -> about 0.38)
    stagnation_threshold: float = 0.1  # Improvement below this value indicates stagnation (eV).
    stagnation_boost: float = 0.2  # Exploration-weight increase during stagnation
    target: float = 0.0  # Target hydrogen adsorption free energy


def compute_adaptive_alpha(
    iteration: int,
    n_samples: int,
    recent_improvement: float | None = None,
    config: AcquisitionConfig | None = None,
) -> dict:
    """Compute adaptive alpha/beta for the current iteration.

    Args:
        iteration: One-based iteration index; iteration 1 is the first round.
        n_samples: Number of samples to select in this round.
        recent_improvement: Previous improvement in best |delta_G_H| (eV),
            or None for the first round.
        config: Acquisition settings.

    Returns:
        dict: alpha, beta, alpha_base, budget_factor, stagnation_boost, etc.
    """
    cfg = config or AcquisitionConfig()

    # 1. Annealing: iteration starts at 1, using alpha_start in the first round.
    alpha_base = cfg.alpha_start * (cfg.decay ** (iteration - 1))

    # 2. Larger sample budgets favor exploration.
    #    n=500 gives a higher factor than n=50.
    #    Map the budget to a multiplier in [0.85, 1.15].
    budget_factor = 0.85 + 0.30 * min(n_samples / 500.0, 1.0)

    # 3. Increase exploration if improvement has stalled.
    boost = 0.0
    if recent_improvement is not None and recent_improvement < cfg.stagnation_threshold:
        boost = cfg.stagnation_boost

    alpha = alpha_base * budget_factor + boost
    alpha = float(np.clip(alpha, cfg.alpha_min, cfg.alpha_max))
    beta = 1.0 - alpha

    return {
        "iteration": iteration,
        "alpha": alpha,
        "beta": beta,
        "alpha_base": alpha_base,
        "budget_factor": budget_factor,
        "stagnation_boost": boost,
        "n_samples": n_samples,
    }


def acquisition_score(
    pred_mean: np.ndarray,
    uncertainty: np.ndarray,
    alpha: float,
    target: float = 0.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute acquisition scores.

    score = alpha * unc_norm + beta * (1 - dist_norm), where unc_norm
    and |pred-target| are normalized to [0,1]. Higher scores take priority.

    Args:
        pred_mean: Predicted delta_G_H, shape (N,).
        uncertainty: Predictive uncertainty, shape (N,).
        alpha: Exploration weight.
        target: Target delta_G_H, default 0.
        eps: Numerical-stability constant.

    Returns:
        Acquisition scores of shape (N,).
    """
    beta = 1.0 - alpha

    # Optimization favors predictions closer to the target.
    dist = np.abs(pred_mean - target)
    dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + eps)
    obj_optimize = 1.0 - dist_norm

    # Exploration favors greater uncertainty.
    unc_norm = (uncertainty - uncertainty.min()) / (
        uncertainty.max() - uncertainty.min() + eps
    )

    return alpha * unc_norm + beta * obj_optimize


def select_samples(
    pred_mean: np.ndarray,
    uncertainty: np.ndarray,
    n_select: int,
    alpha: float,
    exclude_mask: np.ndarray | None = None,
    target: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the top-n samples by acquisition score.

    Args:
        pred_mean: Full-space predictions, shape (N,).
        uncertainty: Full-space uncertainty, shape (N,).
        n_select: Number of samples to select.
        alpha: Exploration weight.
        exclude_mask: Boolean (N,) mask; True excludes a previously computed sample.
        target: Target delta_G_H.

    Returns:
        Selected indices and acquisition scores for all samples.
    """
    scores = acquisition_score(pred_mean, uncertainty, alpha, target=target)

    scores_avail = scores.copy()
    if exclude_mask is not None:
        scores_avail[exclude_mask] = -np.inf

    # Top-n in descending order
    top = np.argsort(scores_avail)[-n_select:][::-1]
    return top, scores
