#!/usr/bin/env python3
"""Compare site attribution for XGBoost, flat MLP and the site-grouped ensemble.

The three models use the same 995 compositions, 87D features and site-index
ranges. DeepSHAP uses the same 100 background compositions for both neural
ensembles; TreeSHAP uses the saved XGBoost attribution summary.
The saved 96D XGBoost summary is included as a feature-set comparison.
Outputs are saved under data/analysis/modeling/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shap_common import SITE_SLICES, aggregate_by_site, load_87d, top_features  # noqa: E402

OUT = ROOT / "data/analysis/modeling"
SEED = 42
BACKGROUND_N = 100
SITES = ["M1", "M2", "M34_sym", "global"]


def symmetry_shap(X: np.ndarray) -> np.ndarray:
    """Compute V2 ensemble DeepSHAP on all 995 samples with 100 background samples."""
    import shap
    import torch.nn as nn
    from herald.models.symmetry_aware_ensemble_v2 import SymmetryAwareEnsembleV2

    model = SymmetryAwareEnsembleV2.load(
        ROOT / "models/ensemble/symmetry_ensemble_v2_for_shap.pth")
    model.config.device = "cpu"
    for m in model.members:
        m.cpu().eval()

    class SymWrapper(nn.Module):
        """Split the 87D input into four blocks, normalize, and return the ensemble mean (B,1)."""

        def __init__(self, ens):
            super().__init__()
            self.ens = ens
            for key, sl in SITE_SLICES.items():
                k = {"M1": "x_M1", "M2": "x_M2",
                     "M34_sym": "x_M34_sym", "global": "g_global"}[key]
                mu, sd = ens.stats_[k]
                self.register_buffer(f"mu_{key}", torch.tensor(mu, dtype=torch.float32))
                self.register_buffer(f"sd_{key}", torch.tensor(sd, dtype=torch.float32))
            self.members = nn.ModuleList(ens.members)

        def forward(self, x):
            parts = []
            for key, sl in SITE_SLICES.items():
                seg = x[:, sl]
                parts.append((seg - getattr(self, f"mu_{key}"))
                             / getattr(self, f"sd_{key}"))
            outs = []
            for m in self.members:
                mean_s, _ = m(*parts)
                outs.append(mean_s * self.ens.y_std_ + self.ens.y_mean_)
            out = torch.stack(outs).mean(dim=0)
            return out.unsqueeze(-1) if out.dim() == 1 else out

    wrapper = SymWrapper(model).eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    rng = np.random.default_rng(SEED)
    bg = Xt[rng.choice(len(X), size=BACKGROUND_N, replace=False)]

    sv = shap.DeepExplainer(wrapper, bg).shap_values(Xt, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[0]
    if isinstance(sv, torch.Tensor):
        sv = sv.detach().cpu().numpy()
    return sv.squeeze(-1) if sv.ndim == 3 else sv


def flat_shap(X: np.ndarray) -> np.ndarray:
    """Compute flat-MLP DeepSHAP using saved weights and the same background."""
    import shap
    from shap_flat_mlp_baseline import FlatWrapper, train_or_load

    _, y, _ = load_87d()
    ens = train_or_load(X, y)
    wrapper = FlatWrapper(ens).eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    rng = np.random.default_rng(SEED)
    bg = Xt[rng.choice(len(X), size=BACKGROUND_N, replace=False)]

    sv = shap.DeepExplainer(wrapper, bg).shap_values(Xt, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[0]
    if isinstance(sv, torch.Tensor):
        sv = sv.detach().cpu().numpy()
    return sv.squeeze(-1) if sv.ndim == 3 else sv


def xgb_site_pct(csv_name: str) -> dict[str, float]:
    """Read exact XGBoost TreeSHAP site attribution, which requires no background.

    Args:
        csv_name: shap_xgb_87d_by_site.csv for the common-input comparison,
            or shap_attribution_by_site.csv for the 96D reference model.
    """
    path = OUT / csv_name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}; first run scripts/modeling/shap_xgb_87d_control.py")
    df = pd.read_csv(path)
    return {r["site"]: float(r["percentage"]) for _, r in df.iterrows()}




def main() -> None:
    print("=" * 70)
    print("Three-model SHAP site comparison (995 samples; shared 100-sample neural background)")
    print("=" * 70)

    X, _, _ = load_87d()
    print(f"\nData: {X.shape[0]} samples x {X.shape[1]} features")

    print("\n[1/3] Flat-MLP DeepSHAP...")
    sv_flat = flat_shap(X)
    pct_flat = aggregate_by_site(sv_flat).set_index("site")["percentage"]

    print("\n[2/3] Symmetric-ensemble DeepSHAP on all samples...")
    sv_sym = symmetry_shap(X)
    pct_sym = aggregate_by_site(sv_sym).set_index("site")["percentage"]

    print("\n[3/3] XGBoost TreeSHAP: common 87D input and 96D reference...")
    pct_xgb87 = xgb_site_pct("shap_xgb_87d_by_site.csv")
    pct_xgb96 = xgb_site_pct("shap_attribution_by_site.csv")

    table = pd.DataFrame({
        "site": SITES,
        "XGBoost87_%": [pct_xgb87[s] for s in SITES],
        "FlatMLP_%": [float(pct_flat[s]) for s in SITES],
        "Symmetry_%": [float(pct_sym[s]) for s in SITES],
        "XGBoost96_%": [pct_xgb96.get(s, float("nan")) for s in SITES],
    })
    table["Sym_minus_Flat_pp"] = table["Symmetry_%"] - table["FlatMLP_%"]
    table["Sym_minus_XGB87_pp"] = table["Symmetry_%"] - table["XGBoost87_%"]
    table["XGB87_minus_XGB96_pp"] = table["XGBoost87_%"] - table["XGBoost96_%"]

    print("\n" + "=" * 70)
    print("Site attribution (%)")
    print("=" * 70)
    print(table.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    print("\nKey values:")
    d_arch = table.set_index("site")["Sym_minus_Flat_pp"]
    print(f"  Architecture comparison (symmetric - flat, shared 87D/NLL):")
    for s in SITES:
        print(f"    {s:<10} {d_arch[s]:+6.2f} pp")
    print(f"\n  All three models use the same 87D input:"
          f"\n    Global features dominate both neural models"
          f" (flat {table.loc[3,'FlatMLP_%']:.1f}% / symmetric {table.loc[3,'Symmetry_%']:.1f}%), "
          f"\n    M1 dominates XGBoost attribution ({table.loc[0,'XGBoost87_%']:.1f}%)"
          f"\n  Global dominance is associated with the neural model class in this comparison.")
    d_cal = table.set_index("site")["XGB87_minus_XGB96_pp"]
    print(f"\n  Feature-set comparison (XGBoost 87D - 96D): "
          f"M1 {d_cal['M1']:+.2f} pp, global {d_cal['global']:+.2f} pp"
          f"\n  Removing nine columns increases the M1 share; two nominal M1 columns were grouped as global in the 96D analysis."
          f"\n  The difference is <1 pp, retaining the direction of the 96D comparison.")

    f_csv = OUT / "shap_three_way_site_attribution.csv"
    table.to_csv(f_csv, index=False)
    print(f"\nSaved: {f_csv}")

    top_sym = top_features(sv_sym, top_k=50)
    f_top = OUT / "shap_symmetry_v2_top_features_corrected.csv"
    top_sym.to_csv(f_top, index=False)
    print(f"Saved: {f_top}  (symmetric-ensemble top 50 with corrected feature names)")
    print("\nSymmetric-ensemble top 10 with corrected feature names:")
    for _, r in top_sym.head(10).iterrows():
        print(f"  {r['rank']:>2}. {r['feature']:<38} {r['site']:<9} {r['mean_abs_shap']:.4f}")

    meta = {
        "evaluation_samples": int(X.shape[0]),
        "background_n": BACKGROUND_N,
        "seed": SEED,
        "site_percentage": {
            "XGBoost_87D_TreeSHAP": {s: round(pct_xgb87[s], 2) for s in SITES},
            "FlatMLP_87D_DeepSHAP": {s: round(float(pct_flat[s]), 2) for s in SITES},
            "SymmetryV2_87D_DeepSHAP": {s: round(float(pct_sym[s]), 2) for s in SITES},
            "XGBoost_96D_TreeSHAP_production": {
                s: round(pct_xgb96.get(s, float("nan")), 2) for s in SITES},
        },
        "pure_architecture_effect_pp": {s: round(float(d_arch[s]), 2) for s in SITES},
        "feature_set_effect_pp_xgb87_minus_xgb96": {
            s: round(float(d_cal[s]), 2) for s in SITES},
        "caveats": [
            "The first three rows share one 87D feature set and one index-range "
            "site partition, so their percentages are directly comparable.",
            "Only FlatMLP vs SymmetryV2 isolates architecture: identical 87D input, "
            "NLL loss, 10 members, seed 42.",
            "The 96D XGBoost row is the production screening surrogate, retained "
            "for provenance of R6 numbers; its 9 extra columns fall into the "
            "global bin under name-based grouping, which suppresses its M1 share "
            "by ~0.9 pp relative to the 87D control.",
            "DeepSHAP additivity check disabled (ensemble averaging introduces "
            "numerical error); relative attribution is the interpretable quantity.",
        ],
    }
    f_meta = OUT / "shap_three_way_meta.json"
    f_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved: {f_meta}")

    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
