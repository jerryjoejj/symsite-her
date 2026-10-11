"""Ridge fitting and saved-formula evaluation for the manuscript expression."""
from __future__ import annotations
import numpy as np

SERIES = {**dict.fromkeys("Sc Ti V Cr Mn Fe Co Ni Cu Zn".split(), "3d"),
          **dict.fromkeys("Ru Rh Pd Ag".split(), "4d"),
          **dict.fromkeys("Ir Pt Au".split(), "5d")}
SHARING = ["independent51"]
ALPHAS = [0., 0.1, 1., 10.]
FITS = 0
FIT_LIMIT = 2000

def predict_formula(x: np.ndarray, metals: np.ndarray, record: dict) -> np.ndarray:
    """Reconstruct predictions from saved raw-unit coefficients."""
    if not set(metals) <= set(record["metals"]):
        raise ValueError("Unseen M1 has no fitted intercept")
    metal_position = np.searchsorted(record["metals"], metals)
    group_names = np.asarray(record["slope_groups"])
    assignment = dict(zip(record["metals"], record["metal_slope_groups"]))
    group_position = np.searchsorted(group_names, [assignment[m] for m in metals])
    return (np.asarray(record["intercepts"])[metal_position]
            + np.sum(x * np.asarray(record["slopes"])[group_position], axis=1))


def fit_predict(x: np.ndarray, y: np.ndarray, metals: np.ndarray,
                train: np.ndarray, test: np.ndarray, sharing: str, alpha: float) -> tuple[np.ndarray, dict]:
    """Profile out M1 intercepts and solve the same standardized Ridge problem."""
    global FITS
    FITS += 1
    if FITS > FIT_LIMIT:
        raise RuntimeError("Registered fit budget exceeded")
    names = np.unique(metals[train])
    if sharing not in SHARING or not set(names) <= set(SERIES):
        raise ValueError("Unregistered sharing rule or metal")
    assignments = ["all" if sharing == "shared19" else SERIES[m] if sharing == "series23" else m
                   for m in names]
    slope_groups = np.unique(assignments)
    group_map = dict(zip(names, assignments))
    xc, yc = np.empty_like(x[train], dtype=float), np.empty_like(y[train], dtype=float)
    means, targets = [], []
    for metal in names:
        mask = metals[train] == metal
        mean_x, mean_y = x[train[mask]].mean(axis=0), y[train[mask]].mean()
        xc[mask], yc[mask] = x[train[mask]] - mean_x, y[train[mask]] - mean_y
        means.append(mean_x)
        targets.append(mean_y)
    scale = np.sqrt(np.mean(xc**2, axis=0))
    scale[scale < 1e-12] = 1.
    design = np.zeros((len(train), len(slope_groups) * x.shape[1]))
    for i, group in enumerate(slope_groups):
        mask = np.array([group_map[m] == group for m in metals[train]])
        design[mask, i * x.shape[1]:(i + 1) * x.shape[1]] = xc[mask] / scale
    if alpha == 0:
        beta = np.linalg.lstsq(design, yc, rcond=None)[0]
    else:
        beta = np.linalg.solve(design.T @ design + alpha * np.eye(design.shape[1]), design.T @ yc)
    slopes = beta.reshape(len(slope_groups), x.shape[1]) / scale
    intercepts = [float(target - mean @ slopes[np.searchsorted(slope_groups, group)])
                  for target, mean, group in zip(targets, means, assignments)]
    record = dict(sharing=sharing, alpha=alpha, metals=names.tolist(), slope_groups=slope_groups.tolist(),
                  metal_slope_groups=assignments, slopes=slopes.tolist(), intercepts=intercepts,
                  scale=scale.tolist(), nominal_parameters=len(names) + len(beta),
                  centered_design_rank=int(np.linalg.matrix_rank(design)))
    prediction = predict_formula(x[test], metals[test], record)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    return prediction, record

