"""Flat-MLP ensemble for architecture/loss comparisons with MSE or NLL.

Comparisons with SymmetryAwareEnsembleV2 use the same 87D symmetric
features and ensemble size. Holding the loss fixed compares architectures
and their associated parameter counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FlatMLPConfig:
    """Configuration for the flat-MLP ensemble."""
    in_dim: int = 87
    hidden: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    loss_type: str = "nll"      # "nll" | "mse"
    n_members: int = 10
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 30
    bootstrap: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FlatMLPMember(nn.Module):
    """One flat-MLP member with a single MSE output or mean/log-variance NLL outputs."""

    def __init__(self, cfg: FlatMLPConfig):
        super().__init__()
        self.loss_type = cfg.loss_type
        layers: list[nn.Module] = []
        prev = cfg.in_dim
        for h in cfg.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(),
                       nn.BatchNorm1d(h), nn.Dropout(cfg.dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, 1)
        if cfg.loss_type == "nll":
            self.logvar_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        mean = self.mean_head(h).squeeze(-1)
        if self.loss_type == "nll":
            logvar = torch.clamp(self.logvar_head(h).squeeze(-1), -10.0, 10.0)
            return mean, logvar
        return mean


def _nll(mean, logvar, target):
    precision = torch.exp(-logvar)
    return 0.5 * (logvar + precision * (target - mean) ** 2).mean()


def _mse(mean, target):
    return ((mean - target) ** 2).mean()


class FlatMLPEnsemble:
    """Flat-MLP deep ensemble with bootstrap sampling and early stopping."""

    def __init__(self, config: FlatMLPConfig | None = None):
        self.config = config or FlatMLPConfig()
        self.members: list[FlatMLPMember] = []
        self.stats_: tuple[np.ndarray, np.ndarray] | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0

    def fit(self, X, y, X_val=None, y_val=None):
        cfg = self.config
        device = torch.device(cfg.device)

        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y) + 1e-8)
        ys = (y - self.y_mean_) / self.y_std_

        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-8
        self.stats_ = (mu, sd)
        Xs = (X - mu) / sd

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv = torch.tensor((X_val - mu) / sd, dtype=torch.float32, device=device)
            yv = torch.tensor((y_val - self.y_mean_) / self.y_std_,
                              dtype=torch.float32, device=device)

        n = len(y)
        rng = np.random.default_rng(cfg.seed)
        self.members = []

        for m in range(cfg.n_members):
            torch.manual_seed(cfg.seed + m)
            model = FlatMLPMember(cfg).to(device)

            idx = rng.integers(0, n, size=n) if cfg.bootstrap else np.arange(n)
            Xb = torch.tensor(Xs[idx], dtype=torch.float32)
            yb = torch.tensor(ys[idx], dtype=torch.float32)
            loader = DataLoader(TensorDataset(Xb, yb),
                                batch_size=cfg.batch_size, shuffle=True)

            optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
            best_val, best_state, patience = float("inf"), None, 0

            for _ in range(cfg.epochs):
                model.train()
                for xb, yb_ in loader:
                    xb, yb_ = xb.to(device), yb_.to(device)
                    optim.zero_grad()
                    out = model(xb)
                    if cfg.loss_type == "nll":
                        loss = _nll(out[0], out[1], yb_)
                    else:
                        loss = _mse(out, yb_)
                    loss.backward()
                    optim.step()

                if has_val:
                    model.eval()
                    with torch.no_grad():
                        out = model(Xv)
                        vloss = (_nll(out[0], out[1], yv).item()
                                 if cfg.loss_type == "nll"
                                 else _mse(out, yv).item())
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
    def predict(self, X) -> np.ndarray:
        cfg = self.config
        device = torch.device(cfg.device)
        mu, sd = self.stats_
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
        means = []
        for model in self.members:
            model.eval()
            out = model(Xt)
            mean_s = out[0] if cfg.loss_type == "nll" else out
            means.append(mean_s.cpu().numpy() * self.y_std_ + self.y_mean_)
        return np.stack(means).mean(axis=0)
