"""Stopping criteria for the active-learning loop.

Convergence requires both small improvements in best |delta_G_H| and few
new qualifying samples over the configured patience window. A cumulative
sample budget supplies a separate stopping condition. Reaching target_dg
is a performance milestone; record_only determines whether it triggers a stop.

The criteria are combined with OR after the minimum-iteration requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class TerminationConfig:
    """Configuration for convergence, budget and performance stopping criteria."""

    improvement_threshold: float = 0.02  # Improvement in best |delta_G_H| below this threshold counts as no progress (eV).
    patience: int = 4                     # Required number of consecutive rounds without progress
    excellent_thr: float = 0.2            # Qualifying threshold for |delta_G_H| (eV)
    min_new_excellent: int = 10           # Fewer new qualifying samples per round indicates diminishing returns.
    min_iterations: int = 4               # Minimum number of iterations before applying stopping criteria
    max_samples: int | None = None        # Cumulative sample limit; None means unlimited.
    target_dg: float = 0.05               # Performance milestone for |delta_G_H|
    target_record_only: bool = True       # Record the performance milestone without stopping.


def _iter_stats(csv_path: Path, excellent_thr: float) -> dict | None:
    """Return best |delta_G_H| and qualifying-sample count for one cumulative set, or None if absent."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "delta_G_H" not in df.columns or len(df) == 0:
        return None
    abs_dg = df["delta_G_H"].abs()
    return {
        "n_samples": len(df),
        "best_abs_dg": float(abs_dg.min()),
        "n_excellent": int((abs_dg <= excellent_thr).sum()),
    }


def build_learning_curve(
    cumulative_dir: str | Path,
    max_iter: int,
    config: TerminationConfig | None = None,
) -> pd.DataFrame:
    """Build per-round best free energies, qualifying counts and increments.

    Args:
        cumulative_dir: Directory containing training_set_iter{k}.csv.
        max_iter: Last iteration to check, from iteration 0 onward.
        config: Stopping settings.

    Returns:
        DataFrame with iteration, n_samples, best_abs_dg, n_excellent,
        improvement from the previous best and new_excellent count.
    """
    cfg = config or TerminationConfig()
    cumulative_dir = Path(cumulative_dir)

    rows = []
    prev_best = None
    prev_exc = None
    for k in range(0, max_iter + 1):
        st = _iter_stats(cumulative_dir / f"training_set_iter{k}.csv", cfg.excellent_thr)
        if st is None:
            continue
        improvement = None if prev_best is None else prev_best - st["best_abs_dg"]
        new_exc = None if prev_exc is None else st["n_excellent"] - prev_exc
        rows.append({
            "iteration": k,
            "n_samples": st["n_samples"],
            "best_abs_dg": st["best_abs_dg"],
            "n_excellent": st["n_excellent"],
            "improvement": improvement,
            "new_excellent": new_exc,
        })
        prev_best = st["best_abs_dg"]
        prev_exc = st["n_excellent"]

    return pd.DataFrame(rows)


def evaluate_termination(
    cumulative_dir: str | Path,
    current_iter: int,
    config: TerminationConfig | None = None,
) -> dict:
    """Evaluate whether the active-learning loop should stop.

    Args:
        cumulative_dir: Directory containing training_set_iter{k}.csv.
        current_iter: Latest completed iteration number.
        config: Stopping settings.

    Returns:
        Dictionary containing should_stop, triggered reasons, detailed
        criterion status, the learning-curve DataFrame and a recommendation.
    """
    cfg = config or TerminationConfig()
    curve = build_learning_curve(cumulative_dir, current_iter, cfg)

    reasons: list[str] = []
    criteria: dict = {}

    if len(curve) == 0:
        return {
            "should_stop": False,
            "reasons": [],
            "criteria": {},
            "learning_curve": curve,
            "recommendation": "No training sets are available for evaluation.",
        }

    latest = curve.iloc[-1]

    # Check the minimum number of iterations.
    if hasattr(cfg, 'min_iterations') and cfg.min_iterations is not None:
        if current_iter < cfg.min_iterations:
            return {
                "should_stop": False,
                "reasons": [],
                "criteria": {"min_iterations_required": True},
                "learning_curve": curve,
                "recommendation": (
                    f"Continue: iteration {current_iter}, "
                    f" has not reached the required minimum ({cfg.min_iterations})."
                    f"The configured minimum requires at least {cfg.min_iterations} iteration points for the learning curve."
                ),
            }

    # Criterion 1: convergence
    # Require at least patience rounds with recorded improvements.
    imp_rows = curve[curve["improvement"].notna()]
    recent = imp_rows.tail(cfg.patience)
    converged = False
    if len(recent) >= cfg.patience:
        stalled_impr = (recent["improvement"].abs() < cfg.improvement_threshold).all()
        # New-qualifying counts must also be available.
        exc_rows = recent[recent["new_excellent"].notna()]
        stalled_exc = (
            len(exc_rows) >= cfg.patience
            and (exc_rows["new_excellent"] < cfg.min_new_excellent).all()
        )
        converged = bool(stalled_impr and stalled_exc)
    criteria["convergence"] = {
        "converged": converged,
        "patience": cfg.patience,
        "recent_improvements": recent["improvement"].round(4).tolist(),
        "recent_new_excellent": recent["new_excellent"].tolist(),
        "improvement_threshold": cfg.improvement_threshold,
        "min_new_excellent": cfg.min_new_excellent,
    }
    if converged:
        reasons.append(
            f"Convergence: {cfg.patience} consecutive rounds with best |delta_G_H| improvement < {cfg.improvement_threshold} "
            f" and new qualifying samples < {cfg.min_new_excellent}"
        )

    # Criterion 2: cumulative sample budget
    budget_hit = False
    if cfg.max_samples is not None:
        budget_hit = int(latest["n_samples"]) >= cfg.max_samples

    criteria["budget"] = {
        "hit": budget_hit,
        "n_samples": int(latest["n_samples"]),
        "max_samples": cfg.max_samples if cfg.max_samples is not None else "unlimited",
    }
    if budget_hit:
        reasons.append(f"Budget reached: cumulative {int(latest['n_samples'])} ≥ {cfg.max_samples}")

    # Criterion 3: performance milestone
    target_hit = float(latest["best_abs_dg"]) < cfg.target_dg
    criteria["performance"] = {
        "hit": target_hit,
        "best_abs_dg": float(latest["best_abs_dg"]),
        "target_dg": cfg.target_dg,
        "record_only": cfg.target_record_only,
    }
    if target_hit and not cfg.target_record_only:
        reasons.append(f"Performance target reached: best |delta_G_H|={latest['best_abs_dg']:.4f} < {cfg.target_dg}")

    should_stop = len(reasons) > 0

    # Human-readable recommendation
    if should_stop:
        rec = "Stop the loop. Triggered criteria: " + "; ".join(reasons)
    else:
        parts = []
        if not converged:
            if len(recent) < cfg.patience:
                parts.append(f"Fewer than {cfg.patience} rounds of improvement records; collect more rounds")
            else:
                parts.append("Not converged: improvement or new qualifying samples remain")
        if not budget_hit:
            if cfg.max_samples is not None:
                parts.append(f"Remaining sample budget: {cfg.max_samples - int(latest['n_samples'])} samples")
            else:
                parts.append("No sample-budget limit; convergence determines stopping")
        rec = "Continue with the next acquisition round." + " ".join(parts)
        if target_hit and cfg.target_record_only:
            rec += f" [Milestone: best |delta_G_H|={latest['best_abs_dg']:.4f} reached; record_only keeps the loop running]"

    return {
        "should_stop": should_stop,
        "reasons": reasons,
        "criteria": criteria,
        "learning_curve": curve,
        "recommendation": rec,
    }
