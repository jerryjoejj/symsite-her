#!/usr/bin/env python3
"""2x2 comparison of architecture and loss on the same 87D input.

The four cells are Flat+MSE, Flat+NLL, Sym+MSE and Sym+NLL (V2).
Compare Sym+NLL with Flat+NLL and Sym+MSE with Flat+MSE to assess
architecture while keeping the input and loss fixed. Compare losses
within each architecture to assess the effect of Gaussian NLL.
Architecture comparisons also change parameter count.

CPU execution uses up to five processes with four threads each.
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from multiprocessing import get_context

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TRAIN_FEATURES = ROOT / "data/processed/training/consolidated_ground_state_features_v2.csv"
OUT = ROOT / "data/analysis/modeling"
N_FOLDS = 5
SEED = 42
THREADS = 4

# Four comparison cells: (architecture, loss)
CELLS = [
    ("Flat_MSE", "flat", "mse"),
    ("Flat_NLL", "flat", "nll"),
    ("Sym_MSE", "sym", "mse"),
    ("Sym_NLL", "sym", "nll"),   # = V2
]


def run_cell(args):
    """Train and evaluate one comparison cell with five-fold CV in a CPU worker."""
    name, arch, loss = args

    import torch
    torch.set_num_threads(THREADS)
    from sklearn.metrics import mean_absolute_error, r2_score
    from herald.features.calculator import (
        FeatureCalculator, load_element_properties)
    from herald.models.symmetry_aware_ensemble_v2 import (
        SymmetryAwareEnsembleV2, SymmetryEnsembleConfigV2)
    from herald.models.flat_mlp_ensemble import FlatMLPEnsemble, FlatMLPConfig

    feat = pd.read_csv(TRAIN_FEATURES)
    y = feat['delta_G_H'].values
    props = load_element_properties(ROOT / "data/elements/element_properties.json")
    calc = FeatureCalculator(props)
    data = calc.compute_symmetric_input_from_features(feat)  # dict of 4 arrays

    # Concatenate the symmetric input blocks into an 87D flat input.
    X_flat = np.concatenate(
        [data['x_M1'], data['x_M2'], data['x_M34_sym'], data['g_global']], axis=1)

    print(f"[{name}] starting (arch={arch}, loss={loss})...", flush=True)

    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    maes, r2s = [], []

    for tr, va in kf.split(y):
        if arch == "sym":
            cfg = SymmetryEnsembleConfigV2(
                n_members=10, epochs=200, early_stop_patience=30, device='cpu')
            # The symmetric model uses NLL by default; switch the loss for MSE.
            ens = SymmetryAwareEnsembleV2(cfg)
            tr_d = {k: v[tr] for k, v in data.items()}
            va_d = {k: v[va] for k, v in data.items()}
            if loss == "mse":
                _fit_sym_mse(ens, tr_d, y[tr], va_d, y[va])
            else:
                ens.fit(tr_d, y[tr], val_data=va_d, y_val=y[va], verbose=False)
            pred = ens.predict(va_d).mean
        else:  # flat
            cfg = FlatMLPConfig(
                in_dim=X_flat.shape[1], loss_type=loss,
                n_members=10, epochs=200, dropout=0.1,
                weight_decay=1e-5, device='cpu')
            ens = FlatMLPEnsemble(cfg)
            ens.fit(X_flat[tr], y[tr], X_val=X_flat[va], y_val=y[va])
            pred = ens.predict(X_flat[va])

        maes.append(mean_absolute_error(y[va], pred))
        r2s.append(r2_score(y[va], pred))

    result = {
        'cell': name, 'arch': arch, 'loss': loss,
        'mae_mean': float(np.mean(maes)), 'mae_std': float(np.std(maes)),
        'r2_mean': float(np.mean(r2s)),
        'maes': [float(m) for m in maes],
    }
    print(f"[{name}] finished: MAE={result['mae_mean']:.4f}±{result['mae_std']:.4f}",
          flush=True)
    return result


def _fit_sym_mse(ens, tr_d, y_tr, va_d, y_va):
    """Train the symmetric architecture with MSE in place of NLL."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from herald.models.symmetry_aware_ensemble_v2 import SymmetryAwareMLP_V2

    cfg = ens.config
    device = torch.device('cpu')
    ens.y_mean_ = float(np.mean(y_tr))
    ens.y_std_ = float(np.std(y_tr) + 1e-8)
    ys = (y_tr - ens.y_mean_) / ens.y_std_
    ds = ens._standardize(tr_d, fit=True)
    dvs = ens._standardize(va_d, fit=False)
    keys = ["x_M1", "x_M2", "x_M34_sym", "g_global"]
    dvs_t = {k: torch.tensor(dvs[k], dtype=torch.float32) for k in keys}
    yv_t = torch.tensor((y_va - ens.y_mean_) / ens.y_std_, dtype=torch.float32)

    n = len(y_tr)
    rng = np.random.default_rng(cfg.seed)
    ens.members = []

    for m in range(cfg.n_members):
        torch.manual_seed(cfg.seed + m)
        model = SymmetryAwareMLP_V2(cfg).to(device)
        idx = rng.integers(0, n, size=n)
        tensors = [torch.tensor(ds[k][idx], dtype=torch.float32) for k in keys]
        yt = torch.tensor(ys[idx], dtype=torch.float32)
        loader = DataLoader(TensorDataset(*tensors, yt),
                            batch_size=cfg.batch_size, shuffle=True)
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
        best_val, best_state, patience = float("inf"), None, 0
        for _ in range(cfg.epochs):
            model.train()
            for batch in loader:
                *xb, yb = [t for t in batch]
                optim.zero_grad()
                mean, _ = model(*xb)
                loss = ((mean - yb) ** 2).mean()  # MSE ignores the log-variance output.
                loss.backward()
                optim.step()
            model.eval()
            with torch.no_grad():
                vm, _ = model(*[dvs_t[k] for k in keys])
                vloss = ((vm - yv_t) ** 2).mean().item()
            if vloss < best_val:
                best_val, patience = vloss, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        ens.members.append(model)
    return ens


def main():
    import time
    from scipy import stats
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("2x2 comparison: architecture x loss, using the same 87D input")
    print("=" * 70)
    print(f"\nStarting {len(CELLS)} cells in parallel (CPU, {THREADS} threads each)...")

    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(processes=len(CELLS)) as pool:
        results = pool.map(run_cell, CELLS)
    print(f"\nAll cells finished in {time.time()-t0:.0f} seconds")

    df = pd.DataFrame([{k: r[k] for k in
                        ['cell', 'arch', 'loss', 'mae_mean', 'mae_std', 'r2_mean']}
                       for r in results])

    # Save fold-level MAEs for paired t tests and two-factor ANOVA.
    # Mean and standard deviation alone do not preserve the paired observations.
    long = pd.DataFrame([
        {'cell': r['cell'], 'arch': r['arch'], 'loss': r['loss'],
         'fold': i, 'mae': m}
        for r in results for i, m in enumerate(r['maes'])])
    long.to_csv(OUT / "ablation_strict_2x2_folds.csv", index=False)
    print(f"Saved fold MAEs: {OUT}/ablation_strict_2x2_folds.csv")
    by = {r['cell']: r for r in results}

    print("\n" + "=" * 70)
    print("2x2 comparison matrix (MAE, eV)")
    print("=" * 70)
    print(f"{'':12s} {'MSE loss':>18s} {'NLL loss':>18s}")
    for arch, label in [("flat", "Flat architecture"), ("sym", "Symmetric architecture")]:
        mse = by[f"{'Flat' if arch=='flat' else 'Sym'}_MSE"]
        nll = by[f"{'Flat' if arch=='flat' else 'Sym'}_NLL"]
        print(f"{label:12s} {mse['mae_mean']:.4f}±{mse['mae_std']:.4f}    "
              f"{nll['mae_mean']:.4f}±{nll['mae_std']:.4f}")

    # Paired t tests for architecture and loss comparisons
    print("\n" + "=" * 70)
    print("Architecture and loss comparisons (paired by CV fold)")
    print("=" * 70)

    def paired(a, b, desc):
        ta, tb = by[a]['maes'], by[b]['maes']
        t, p = stats.ttest_rel(ta, tb)
        d = np.mean(ta) - np.mean(tb)
        sig = "significant" if p < 0.05 else "not significant"
        print(f"{desc}: ΔMAE={d:+.4f}, t={t:.2f}, p={p:.3f} [{sig}]")
        return d, p

    print("\n--- Architecture comparison: fixed input and loss ---")
    paired("Sym_NLL", "Flat_NLL", "Sym vs Flat @ NLL")
    paired("Sym_MSE", "Flat_MSE", "Sym vs Flat @ MSE")
    print("\n--- Loss comparison: fixed input and architecture ---")
    paired("Sym_NLL", "Sym_MSE", "NLL vs MSE @ Sym")
    paired("Flat_NLL", "Flat_MSE", "NLL vs MSE @ Flat")

    df.to_csv(OUT / "ablation_strict_2x2.csv", index=False)
    print(f"\nSaved: {OUT}/ablation_strict_2x2.csv")


if __name__ == "__main__":
    main()
