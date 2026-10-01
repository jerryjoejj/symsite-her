#!/usr/bin/env python3
"""Compute DeepSHAP attribution for the flat-MLP comparison.

The flat and site-grouped ensembles share the 87D input, Gaussian NLL,
ten members and random seed. Their architectures and parameter counts differ.
The script loads the supplied flat-MLP weights or trains them when absent,
then writes site attributions and feature rankings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shap_common import (  # noqa: E402
    aggregate_by_site, load_87d, top_features,
)
from herald.models.flat_mlp_ensemble import FlatMLPConfig, FlatMLPEnsemble  # noqa: E402

OUT = ROOT / "data/analysis/modeling"
MODEL_PATH = ROOT / "models/ensemble/flat_mlp_nll_for_shap.pt"
SEED = 42
BACKGROUND_N = 100


class FlatWrapper(nn.Module):
    """Wrap FlatMLPEnsemble as a single-input SHAP-compatible nn.Module.

    Forward passes run on CPU and retain the gradient graph. Inputs are
    unscaled 87D features; normalization occurs inside the wrapper.
    Return the ensemble mean in eV with shape (B, 1).
    """

    def __init__(self, ens: FlatMLPEnsemble):
        super().__init__()
        mu, sd = ens.stats_
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.y_mean = float(ens.y_mean_)
        self.y_std = float(ens.y_std_)
        self.members = nn.ModuleList([m.cpu().eval() for m in ens.members])
        self.loss_type = ens.config.loss_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = (x - self.mu) / self.sd
        outs = []
        for m in self.members:
            o = m(xs)
            mean_s = o[0] if self.loss_type == "nll" else o
            outs.append(mean_s * self.y_std + self.y_mean)
        out = torch.stack(outs).mean(dim=0)
        return out.unsqueeze(-1) if out.dim() == 1 else out


def train_or_load(X: np.ndarray, y: np.ndarray) -> FlatMLPEnsemble:
    """Load saved flat-ensemble weights, or train on CPU with the 2x2 comparison settings."""
    cfg = FlatMLPConfig(
        in_dim=X.shape[1], loss_type="nll", n_members=10, epochs=200,
        dropout=0.1, weight_decay=1e-5, early_stop_patience=30,
        bootstrap=True, seed=SEED, device="cpu",
    )
    ens = FlatMLPEnsemble(cfg)

    if MODEL_PATH.exists():
        ckpt = torch.load(MODEL_PATH, weights_only=False)
        ens.stats_ = ckpt["stats"]
        ens.y_mean_, ens.y_std_ = ckpt["y_mean"], ckpt["y_std"]
        from herald.models.flat_mlp_ensemble import FlatMLPMember
        ens.members = []
        for state in ckpt["member_states"]:
            m = FlatMLPMember(cfg)
            m.load_state_dict(state)
            m.eval()
            ens.members.append(m)
        print(f"  Loading saved weights: {MODEL_PATH.name}")
        return ens

    # Use the same 80/20 split and seed as train_symmetry_ensemble_v2_for_shap.py.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    tr, va = idx[:split], idx[split:]
    print(f"  Training set {len(tr)} / validation set {len(va)}, ten NLL members, CPU")
    ens.fit(X[tr], y[tr], X_val=X[va], y_val=y[va])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": cfg,
        "stats": ens.stats_,
        "y_mean": ens.y_mean_,
        "y_std": ens.y_std_,
        "member_states": [m.state_dict() for m in ens.members],
    }, MODEL_PATH)
    print(f"  Weights saved: {MODEL_PATH}")

    from sklearn.metrics import mean_absolute_error, r2_score
    pred = ens.predict(X[va])
    print(f"  Validation MAE={mean_absolute_error(y[va], pred):.4f} eV  "
          f"R²={r2_score(y[va], pred):.4f}")
    return ens


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Flat-MLP SHAP attribution (Flat_NLL cell of the 2x2 comparison)")
    print("=" * 70)

    print("\n[1/4] Loading 87D symmetric features...")
    X, y, _ = load_87d()
    print(f"  {X.shape[0]} samples x {X.shape[1]} features")

    print("\n[2/4] Training/loading the flat ensemble...")
    ens = train_or_load(X, y)

    print("\n[3/4] DeepSHAP on CPU with 100 background samples...")
    wrapper = FlatWrapper(ens).eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    rng = np.random.default_rng(SEED)
    bg = Xt[rng.choice(len(X), size=BACKGROUND_N, replace=False)]

    explainer = shap.DeepExplainer(wrapper, bg)
    # Match the symmetric ensemble: interpret relative (N,87) attributions with additivity checking disabled.
    sv = explainer.shap_values(Xt, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[0]
    if isinstance(sv, torch.Tensor):
        sv = sv.detach().cpu().numpy()
    if sv.ndim == 3:
        sv = sv.squeeze(-1)
    print(f"  SHAP matrix {sv.shape}")

    print("\n[4/4] Aggregating and saving...")
    by_site = aggregate_by_site(sv)
    print("\nSite attribution for the flat MLP on all 995 samples:")
    for _, r in by_site.iterrows():
        print(f"  {r['site']:<10} {r['percentage']:>6.1f}%  "
              f"(total mean|SHAP| {r['mean_abs_shap']:.4f})")

    f_site = OUT / "shap_flat_mlp_by_site.csv"
    by_site.to_csv(f_site, index=False)
    print(f"\n  Saved: {f_site}")

    top = top_features(sv, top_k=50)
    print("\nTop 10 features:")
    for _, r in top.head(10).iterrows():
        print(f"  {r['rank']:>2}. {r['feature']:<38} {r['site']:<9} "
              f"{r['mean_abs_shap']:.4f}")

    f_top = OUT / "shap_flat_mlp_top_features.csv"
    top.to_csv(f_top, index=False)
    print(f"\n  Saved: {f_top}")

    meta = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "background_n": BACKGROUND_N,
        "seed": SEED,
        "architecture": "FlatMLPEnsemble(hidden=(256,128,64), loss=nll, n=10)",
        "role": "2x2 ablation Flat_NLL cell — same 87D input & NLL loss as "
                "SymmetryAwareEnsembleV2, architecture is the only difference",
        "site_percentage": {r["site"]: round(float(r["percentage"]), 2)
                            for _, r in by_site.iterrows()},
    }
    f_meta = OUT / "shap_flat_mlp_meta.json"
    f_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  Saved: {f_meta}")

    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
