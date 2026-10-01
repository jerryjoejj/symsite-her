"""Parse saved DFT energies and calculate hydrogen adsorption free energies.

Read output.txt, pair clean/H energies and apply:
    delta_G_H = E(H_site) - E(clean) + 3.481 eV.

The correction combines -1/2 E(H2), zero-point energy and entropy terms.
Negative delta_G_H denotes stronger hydrogen binding; positive values
denote weaker binding, with near-zero values defining the activity target.

Input examples:
    001_PtAuFeAg : energy  without entropy=  -598.78  energy(sigma->0) =  -598.79
    001_PtAuFeAg_H : energy  without entropy=  -602.49  ...
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Shared adsorption-free-energy correction in eV
DELTA_G_CORRECTION = 3.481


def parse_output(output_file: str | Path) -> dict[str, float]:
    """Parse energy-without-entropy values from readentropy.py output.txt.

    Args:
        output_file: Path to output.txt.

    Returns:
        Mapping from structure name to energy, e.g. {"001_PtAuFeAg": -598.78}.
    """
    output_file = Path(output_file)
    energies: dict[str, float] = {}

    pattern = re.compile(r"energy\s+without\s+entropy=\s*(-?\d+\.?\d*)")

    with open(output_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if " : " not in line:
                continue
            name, rest = line.split(" : ", 1)
            m = pattern.search(rest)
            if m:
                energies[name.strip()] = float(m.group(1))

    return energies


def compute_delta_g(
    energies: dict[str, float],
    structures: pd.DataFrame,
    correction: float = DELTA_G_CORRECTION,
) -> pd.DataFrame:
    """Compute delta_G_H for each sample from paired total energies.

    Args:
        energies: Output of parse_output.
        structures: DataFrame with M1..M4 in sample-ID order.
        correction: Free-energy correction, default 3.481 eV.

    Returns:
        DataFrame with sample_id, M1-M4, E_clean, E_H, delta_E_H, delta_G_H and status.
    """
    rows = []
    for idx, row in structures.iterrows():
        sample_id = f"{idx + 1:03d}"
        alloy = f"{row['M1']}{row['M2']}{row['M3']}{row['M4']}"
        clean_name = f"{sample_id}_{alloy}"
        h_name = f"{sample_id}_{alloy}_H"

        rec = {
            "sample_id": sample_id,
            "M1": row["M1"], "M2": row["M2"],
            "M3": row["M3"], "M4": row["M4"],
        }
        if "index" in structures.columns:
            rec["index"] = row["index"]

        e_clean = energies.get(clean_name)
        e_h = energies.get(h_name)

        if e_clean is None:
            rec.update(E_clean=None, E_H=None, delta_E_H=None,
                       delta_G_H=None, status="missing_clean")
        elif e_h is None:
            rec.update(E_clean=e_clean, E_H=None, delta_E_H=None,
                       delta_G_H=None, status="missing_H")
        else:
            d_e = e_h - e_clean
            rec.update(
                E_clean=e_clean, E_H=e_h, delta_E_H=d_e,
                delta_G_H=d_e + correction, status="success",
            )
        rows.append(rec)

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    """Summarize adsorption-free-energy statistics."""
    ok = results[results["status"] == "success"]
    if len(ok) == 0:
        return {"n_total": len(results), "n_success": 0}

    dg = ok["delta_G_H"]
    abs_dg = dg.abs()
    best = ok.loc[abs_dg.idxmin()]
    return {
        "n_total": len(results),
        "n_success": len(ok),
        "success_rate": len(ok) / len(results),
        "dg_min": float(dg.min()),
        "dg_max": float(dg.max()),
        "dg_mean": float(dg.mean()),
        "dg_std": float(dg.std()),
        "n_excellent": int((abs_dg <= 0.2).sum()),
        "best_sample": f"{best['M1']}-{best['M2']}-{best['M3']}-{best['M4']}",
        "best_dg": float(best["delta_G_H"]),
    }
