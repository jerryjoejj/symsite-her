"""Site-role components and evaluation of the manuscript's two representations."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import parameter_sharing as engine

ROOT = Path(__file__).resolve().parents[2]

def macro_mae(y, pred, metals):
    return float(np.mean([np.mean(np.abs(y[metals == m] - pred[metals == m])) for m in np.unique(metals)]))

def role_axes(radius: np.ndarray, electrons: np.ndarray) -> dict[str, np.ndarray]:
    """Average nonlinear axes over declared near-site assignments, preserving M34."""
    if radius.shape != electrons.shape or radius.ndim != 2 or radius.shape[1] != 4:
        raise ValueError("Expected aligned M1/M2/M3/M4 arrays")
    by_near = []
    for near in [1, 2, 3]:
        far = [i for i in [1, 2, 3] if i != near]
        dr = (
            radius[:, near]
            + radius[:, far].mean(axis=1)
            + abs(radius[:, far[0]] - radius[:, far[1]])
            - 2 * radius[:, 0]
        ) / 100
        dn = electrons[:, near] - electrons[:, 0]
        de1 = dn * (electrons[:, far].mean(axis=1) - electrons[:, near])
        by_near.append(np.column_stack([dr, de1, dn**2]))
    stacked = np.stack(by_near)
    values = dict(
        original=stacked[0],
        unordered=stacked.mean(axis=0),
        swapped=stacked[1:].mean(axis=0),
    )
    return {
        family + "_" + variant: array[:, [0, electronic]]
        for family, electronic in [("De1", 1), ("De2", 2)]
        for variant, array in values.items()
    }


def element_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Read immutable elemental properties, without consulting target values."""
    arrays = []
    for property_name in ["atomic_radius", "d_electrons"]:
        groups = frame.groupby("M1")["M1_" + property_name]
        assert (groups.nunique() == 1).all()
        lookup = groups.first()
        array = np.column_stack([frame["M" + str(i)].map(lookup) for i in range(1, 5)])
        assert np.isfinite(array).all()
        arrays.append(array)
    return tuple(arrays)


def evaluate(matrices: dict, y: np.ndarray, metals: np.ndarray, fold: dict) -> tuple:
    """Tune alpha only, separately for every fixed representation, inside training."""
    train, test = np.array(fold["train"]), np.array(fold["test"])
    predictions, models, trials = {}, [], []
    for name, x in matrices.items():
        scores = []
        for alpha in engine.ALPHAS:
            inner = np.full(len(y), np.nan)
            for part in fold["inner"]:
                a, b = np.array(part["train"]), np.array(part["test"])
                inner[b], _ = engine.fit_predict(
                    x, y, metals, a, b, "independent51", alpha
                )
            assert np.isfinite(inner[train]).all()
            scores.append(
                dict(
                    model=name,
                    alpha=alpha,
                    inner_macro_mae=macro_mae(y[train], inner[train], metals[train]),
                )
            )
        chosen = min(scores, key=lambda s: (s["inner_macro_mae"], s["alpha"]))
        predictions[name], formula = engine.fit_predict(
            x, y, metals, train, test, "independent51", chosen["alpha"]
        )
        models.append(dict(chosen, formula=formula))
        trials.extend(scores)
    for name, statistic in [("M1_mean", np.mean), ("M1_median", np.median)]:
        baseline = {
            m: statistic(y[train[metals[train] == m]]) for m in np.unique(metals[train])
        }
        predictions[name] = np.array([baseline[m] for m in metals[test]])
    return predictions, models, trials


def manuscript_axes(frame):
    """Construct the original and unordered D01 components."""
    axes = role_axes(*element_arrays(frame))
    return {name: axes[name] for name in ["De1_original", "De1_unordered"]}


def main():
    """Summarize the supplied fold predictions and check the published metrics."""
    root = ROOT / "data/analysis/descriptors/site_role_controls"
    predictions = pd.read_csv(root / "predictions.csv")
    metrics = pd.read_csv(root / "metrics.csv")
    results = []
    for case, rows in predictions.groupby("case", sort=False):
        for model in ["De1_original", "De1_unordered"]:
            score = macro_mae(rows.delta_G_H.to_numpy(), rows[model].to_numpy(), rows.M1.to_numpy())
            recorded = metrics[(metrics.case == case) & (metrics.model == model)].macro_mae.item()
            assert abs(score - recorded) < 1e-12
            results.append(dict(case=case, model=model, macro_mae_eV=score, n=len(rows)))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
