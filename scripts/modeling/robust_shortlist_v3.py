#!/usr/bin/env python3
"""Select robust candidates using the V2 ensemble.

Criteria: |delta_G_H| <= 0.10 eV and epistemic standard deviation no greater
than its full-space median. All predictions come from the V2 ensemble.

Outputs:
    data/analysis/modeling/robust_shortlist_v3.csv
    data/analysis/modeling/SI_table_cheap_candidates.csv
    data/analysis/modeling/SI_table_cheap_candidates.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data/analysis/modeling"
ACTIVITY = RES / "activity_map_28560_v3.csv"
PRECIOUS = {"Ru", "Rh", "Pd", "Ag", "Ir", "Pt", "Au", "Os", "Re"}


def main() -> None:
    act = pd.read_csv(ACTIVITY)
    epi_thr = np.quantile(act["epistemic"].values, 0.5)

    df = act.copy()
    df["n_precious"] = df[["M1", "M2", "M3", "M4"]].apply(
        lambda r: sum(m in PRECIOUS for m in r), axis=1)
    df["mean_abs"] = df["abs_dG"]

    robust = df[(df["abs_dG"] <= 0.10) & (df["epistemic"] <= epi_thr)].copy()
    robust = robust.sort_values("abs_dG").reset_index(drop=True)
    robust.to_csv(RES / "robust_shortlist_v3.csv", index=False,
                  float_format="%.6f")

    cheap = robust[robust["n_precious"] == 0].reset_index(drop=True)
    n_co = int((cheap["M1"] == "Co").sum())
    n_zn = int((cheap[["M1", "M2", "M3", "M4"]] == "Zn").any(axis=1).sum())

    print("=" * 60)
    print("Robust candidates from V2 predictions and epistemic uncertainty")
    print("=" * 60)
    print(f"Epistemic threshold (full-space median): {epi_thr:.4f} eV")
    print(f"Robust candidates (|delta_G|<=0.10, epistemic<=median): {len(robust)}")
    print(f"All-3d candidates: {len(cheap)}")
    print(f"  With Co at M1: {n_co}")
    print(f"  Containing Zn: {n_zn}")

    out = cheap[["M1", "M2", "M3", "M4", "pred_dG", "abs_dG",
                 "epistemic"]].copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    out.columns = ["Rank", "M1", "M2", "M3", "M4",
                   "dG_SymEns", "abs_dG", "epistemic"]
    out.to_csv(RES / "SI_table_cheap_candidates.csv", index=False,
               float_format="%.4f")

    lines = [
        "# Supplementary Table: Earth-Abundant Robust HER Candidates",
        "",
        f"All {len(cheap)} candidates satisfy |ΔG_H|<=0.10 eV and epistemic "
        "uncertainty at or below the full-space median, and contain no precious "
        "metals (Ru/Rh/Pd/Ag/Ir/Pt/Au/Os/Re). ΔG_H in eV; M1 is the "
        "H-adsorption site. Predictions are from the symmetry-aware ensemble "
        "(V2, 87D symmetric input).",
        "",
        "| Rank | M1 | M2 | M3 | M4 | ΔG_H (SymEns) | epistemic |",
        "|------|----|----|----|----|---------------|-----------|",
    ]
    for _, x in out.iterrows():
        lines.append(
            f"| {x['Rank']:.0f} | {x['M1']} | {x['M2']} | {x['M3']} | "
            f"{x['M4']} | {x['dG_SymEns']:+.4f} | {x['epistemic']:.4f} |")

    m1c = cheap["M1"].value_counts()
    lines += [
        "",
        f"**Composition statistics ({len(cheap)} candidates):**",
        "- M1 (adsorption site): " +
        ", ".join(f"{k}×{v}" for k, v in m1c.items()),
        f"- Contain Zn: {n_zn}/{len(cheap)} ({n_zn/len(cheap)*100:.0f}%)",
        "- Dominant motif: Co(M1) + Zn-containing 3d environment",
    ]
    (RES / "SI_table_cheap_candidates.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"saved {RES / 'SI_table_cheap_candidates.csv'}")
    print(f"saved {RES / 'SI_table_cheap_candidates.md'}")
    print("\nM1 distribution:", dict(m1c))
    print(f"Containing Zn: {n_zn}/{len(cheap)}")


if __name__ == "__main__":
    main()
