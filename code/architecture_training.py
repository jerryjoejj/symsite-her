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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" if (ROOT / "src").exists() else ROOT / "code/src"))

TRAIN_FEATURES = ROOT / "data/processed/training/consolidated_ground_state_features_v2.csv"
OUT = ROOT / "outputs/retraining"
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
