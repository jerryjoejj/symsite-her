"""Architecture variants of the symmetry-aware V2 model.

GlobalEncodedMLP_V2 adds a lightweight encoder for the 52D global block.
M1M2InteractionMLP_V2 adds explicit M1-M2 interactions before fusion.
Both retain the V2 site encoders, 87D input and heteroscedastic NLL.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from herald.models.symmetry_aware_ensemble_v2 import (
    SymmetryEnsembleConfigV2,
    SiteEncoder,
    gaussian_nll_loss,
    SymPrediction,
)


class _EnsembleMixin:
    """Use V2 ensemble fitting and prediction with a configurable member class."""

    member_cls = None

    def __init__(self, config: SymmetryEnsembleConfigV2 | None = None):
        self.config = config or SymmetryEnsembleConfigV2()
        self.members = []
        self.stats_: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _standardize(self, data, fit=False):
        out = {}
        for key, arr in data.items():
            if fit:
                mu = arr.mean(axis=0)
                sd = arr.std(axis=0) + 1e-8
                self.stats_[key] = (mu, sd)
            mu, sd = self.stats_[key]
            out[key] = (arr - mu) / sd
        return out

    def fit(self, data, y, val_data=None, y_val=None, verbose=False):
        cfg = self.config
        device = torch.device(cfg.device)
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y) + 1e-8)
        ys = (y - self.y_mean_) / self.y_std_
        ds = self._standardize(data, fit=True)
        n = len(y)
        rng = np.random.default_rng(cfg.seed)
        keys = ["x_M1", "x_M2", "x_M34_sym", "g_global"]
        has_val = val_data is not None and y_val is not None
        if has_val:
            dvs = self._standardize(val_data, fit=False)
            dvs_t = {k: torch.tensor(v, dtype=torch.float32, device=device)
                     for k, v in dvs.items()}
            yv_t = torch.tensor((y_val - self.y_mean_) / self.y_std_,
                                dtype=torch.float32, device=device)
        self.members = []
        for m in range(cfg.n_members):
            torch.manual_seed(cfg.seed + m)
            model = self.member_cls(cfg).to(device)
            idx = rng.integers(0, n, size=n) if cfg.bootstrap else np.arange(n)
            tensors = [torch.tensor(ds[k][idx], dtype=torch.float32)
                       for k in keys]
            yt = torch.tensor(ys[idx], dtype=torch.float32)
            loader = DataLoader(TensorDataset(*tensors, yt),
                                batch_size=cfg.batch_size, shuffle=True)
            optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
            best_val, best_state, patience = float("inf"), None, 0
            for _ in range(cfg.epochs):
                model.train()
                for batch in loader:
                    *xb, yb = [t.to(device) for t in batch]
                    optim.zero_grad()
                    mean, logvar = model(*xb)
                    loss = gaussian_nll_loss(mean, logvar, yb)
                    loss.backward()
                    optim.step()
                if has_val:
                    model.eval()
                    with torch.no_grad():
                        vm, vlv = model(*[dvs_t[k] for k in keys])
                        vloss = gaussian_nll_loss(vm, vlv, yv_t).item()
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
            self.members.append(model)
        return self

    @torch.no_grad()
    def predict(self, data):
        cfg = self.config
        device = torch.device(cfg.device)
        ds = self._standardize(data, fit=False)
        keys = ["x_M1", "x_M2", "x_M34_sym", "g_global"]
        xt = [torch.tensor(ds[k], dtype=torch.float32, device=device)
              for k in keys]
        means, vars_ = [], []
        for model in self.members:
            model.eval()
            mean_s, logvar_s = model(*xt)
            means.append(mean_s.cpu().numpy() * self.y_std_ + self.y_mean_)
            vars_.append(np.exp(logvar_s.cpu().numpy()) * (self.y_std_ ** 2))
        member_means = np.stack(means)
        return SymPrediction(
            member_means.mean(axis=0),
            np.stack(vars_).mean(axis=0),
            member_means.var(axis=0),
            None,
            member_means)


class GlobalEncodedMLP_V2(nn.Module):
    """Encode global features with Linear(52,64), ReLU, batch normalization and dropout before fusion."""

    def __init__(self, cfg: SymmetryEnsembleConfigV2):
        super().__init__()
        self.enc_M1 = SiteEncoder(cfg.m1_dim, cfg.site_embed, cfg.dropout)
        self.enc_M2 = SiteEncoder(cfg.m2_dim, cfg.site_embed, cfg.dropout)
        self.enc_M34 = SiteEncoder(cfg.m34_sym_dim, cfg.site_embed, cfg.dropout)
        self.enc_global = nn.Sequential(
            nn.Linear(cfg.global_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Dropout(cfg.dropout))
        fusion_in = 3 * cfg.site_embed + 64
        layers = []
        prev = fusion_in
        for h in cfg.fusion_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h),
                       nn.Dropout(cfg.dropout)]
            prev = h
        self.fusion = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, 1)
        self.logvar_head = nn.Linear(prev, 1)

    def forward(self, x_M1, x_M2, x_M34_sym, g_global):
        h1 = self.enc_M1(x_M1)
        h2 = self.enc_M2(x_M2)
        h34 = self.enc_M34(x_M34_sym)
        hg = self.enc_global(g_global)
        h = self.fusion(torch.cat([h1, h2, h34, hg], dim=1))
        return self.mean_head(h).squeeze(-1), \
            torch.clamp(self.logvar_head(h).squeeze(-1), -10.0, 10.0)


class M1M2InteractionMLP_V2(nn.Module):
    """Add the elementwise product of M1 and M2 embeddings to the fusion input."""

    def __init__(self, cfg: SymmetryEnsembleConfigV2):
        super().__init__()
        self.enc_M1 = SiteEncoder(cfg.m1_dim, cfg.site_embed, cfg.dropout)
        self.enc_M2 = SiteEncoder(cfg.m2_dim, cfg.site_embed, cfg.dropout)
        self.enc_M34 = SiteEncoder(cfg.m34_sym_dim, cfg.site_embed, cfg.dropout)
        fusion_in = 3 * cfg.site_embed + cfg.global_dim + cfg.site_embed
        layers = []
        prev = fusion_in
        for h in cfg.fusion_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h),
                       nn.Dropout(cfg.dropout)]
            prev = h
        self.fusion = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, 1)
        self.logvar_head = nn.Linear(prev, 1)

    def forward(self, x_M1, x_M2, x_M34_sym, g_global):
        h1 = self.enc_M1(x_M1)
        h2 = self.enc_M2(x_M2)
        h34 = self.enc_M34(x_M34_sym)
        h12 = h1 * h2
        h = self.fusion(torch.cat([h1, h2, h34, h12, g_global], dim=1))
        return self.mean_head(h).squeeze(-1), \
            torch.clamp(self.logvar_head(h).squeeze(-1), -10.0, 10.0)


class GlobalEncodedEnsembleV2(_EnsembleMixin):
    member_cls = GlobalEncodedMLP_V2


class M1M2InteractionEnsembleV2(_EnsembleMixin):
    member_cls = M1M2InteractionMLP_V2
