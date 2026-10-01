"""Evaluate the manuscript D01 expression from saved coefficients and elemental properties."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def identity(row: dict, elements: dict) -> tuple[str, list[str]]:
    """Require four distinct supported metals and preserve the M1/M2 roles."""
    metals = (
        [row[f"M{i}"] for i in range(1, 5)]
        if all(f"M{i}" in row for i in range(1, 5))
        else row["physical_id"].split("-")
    )
    if len(metals) != 4 or len(set(metals)) != 4 or not set(metals) <= set(elements):
        raise ValueError("Expected four distinct metals from the locked element table")
    pid = "-".join(metals[:2] + sorted(metals[2:]))
    if (
        row.get("physical_id")
        and identity_from_text(row["physical_id"], elements) != pid
    ):
        raise ValueError("physical_id conflicts with site columns")
    return pid, metals


def identity_from_text(text: str, elements: dict) -> str:
    metals = text.split("-")
    if len(metals) != 4 or len(set(metals)) != 4 or not set(metals) <= set(elements):
        raise ValueError("Invalid physical_id")
    return "-".join(metals[:2] + sorted(metals[2:]))


def composition_axes(
    metals: list[str], elements: dict, role: str, family: str
) -> list[float]:
    """Compute the frozen tabulated-property axes, without DFT or model labels."""
    r = [elements[m]["radius_pm"] for m in metals]
    n = [elements[m]["d_electrons"] for m in metals]
    if role not in ["original", "unordered"] or family not in ["De1", "De2"]:
        raise ValueError("Unknown axis definition")
    assignments = [1] if role == "original" else [1, 2, 3]
    values = []
    for near in assignments:
        far = [i for i in [1, 2, 3] if i != near]
        radius = (
            r[near]
            + (r[far[0]] + r[far[1]]) / 2
            + abs(r[far[0]] - r[far[1]])
            - 2 * r[0]
        ) / 100
        dn = n[near] - n[0]
        electronic = (
            dn * ((n[far[0]] + n[far[1]]) / 2 - n[near]) if family == "De1" else dn**2
        )
        values.append([radius, electronic])
    return [sum(v[j] for v in values) / len(values) for j in [0, 1]]


def features(row: dict, model: dict, elements: dict) -> tuple[list[float], list[float]]:
    """Require clean electronic features explicitly; never impute unavailable data."""
    _, metals = identity(row, elements)
    if model["scope"] != "all17" and metals[0] != model["scope"]:
        raise ValueError(f"Model scope is {model['scope']}, not {metals[0]}")
    if model["kind"] == "electronic":
        axes = [float(row[c]) for c in model["columns"]]
    else:
        axes = composition_axes(metals, elements, model["role"], model["family"])
    if not all(math.isfinite(a) for a in axes):
        raise ValueError("Nonfinite input")
    x = [axes[0] ** a * axes[1] ** b for a, b in model["powers"]]
    return axes, x


def predict(row: dict, model: dict, elements: dict) -> dict:
    """Return the signed estimate and empirical support flag, not a probability."""
    pid, metals = identity(row, elements)
    axes, x = features(row, model, elements)
    f = model["formula"]
    position = f["metals"].index(metals[0])
    value = f["intercepts"][position] + sum(
        a * b for a, b in zip(x, f["coefficients"][position])
    )
    domain = model["domains"][metals[0]]
    if len(domain["dimensions"]) == 1:
        q = axes[domain["dimensions"][0]]
        inside = domain["minimum"][0] - 1e-10 <= q <= domain["maximum"][0] + 1e-10
    else:
        inside = all(
            a * axes[0] + b * axes[1] + c <= 1e-10
            for a, b, c in domain["hull_equations"]
        )
    if not math.isfinite(value):
        raise ValueError("Nonfinite prediction")
    return dict(
        physical_id=pid,
        model=model["id"],
        predicted_delta_G_H_eV=value,
        inside_training_hull=inside,
        support_is_not_validation=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    model = next(m for m in lock["models"] if m["id"] == args.model)
    with args.input.open(newline="") as handle:
        rows = [predict(r, model, lock["elements"]) for r in csv.DictReader(handle)]
    if not rows:
        raise ValueError("Empty query file")
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions; {lock['status']}")


if __name__ == "__main__":
    main()
