"""Symmetry-aware heteroscedastic deep ensemble V2.

The 87D input comprises M1 (16), M2 (13), symmetric M34 statistics (6)
and global statistics (52), selected from the 96D feature table.
Independent encoders process M1, M2 and M34; the six remote statistics
are encoded directly from 6 to 32 dimensions. Mean/absolute-difference
inputs make predictions invariant to M3/M4 exchange.
Training uses heteroscedastic Gaussian NLL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Configuration

@dataclass
class SymmetryEnsembleConfigV2:
    """Configuration for the symmetry-aware V2 ensemble."""

    m1_dim: int = 16            # M1 input dimension, including derived features
    m2_dim: int = 13            # M2 input dimension
    m34_sym_dim: int = 6        # Symmetric M34 input dimension
    global_dim: int = 52        # Global-statistics dimension
    site_embed: int = 32        # Site-embedding dimension
    fusion_dims: tuple[int, ...] = (128, 64)  # Fusion layers
    dropout: float = 0.1
    n_members: int = 10         # Number of ensemble members
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 30
    bootstrap: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Base learner

class SiteEncoder(nn.Module):
    """Encode an input block from in_dim to embed dimensions."""

    def __init__(self, in_dim: int, embed: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed),
            nn.ReLU(),
            nn.BatchNorm1d(embed),
            nn.Dropout(dropout),
            nn.Linear(embed, embed),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SymmetryAwareMLP_V2(nn.Module):
    """Symmetry-aware heteroscedastic MLP V2.

    Inputs: M1 (16D), M2 (13D), M34_sym (6D) and global (52D).
    The remote block already contains exchange-invariant statistics.
    """

    def __init__(self, cfg: SymmetryEnsembleConfigV2):
        super().__init__()
        self.cfg = cfg

        # Three independent encoders with different input dimensions
        self.enc_M1 = SiteEncoder(cfg.m1_dim, cfg.site_embed, cfg.dropout)
        self.enc_M2 = SiteEncoder(cfg.m2_dim, cfg.site_embed, cfg.dropout)
        self.enc_M34 = SiteEncoder(cfg.m34_sym_dim, cfg.site_embed, cfg.dropout)

        # Fusion network: 3*embed + global_dim -> fusion_dims
        fusion_in = 3 * cfg.site_embed + cfg.global_dim
        layers: list[nn.Module] = []
        prev = fusion_in
        for h in cfg.fusion_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(),
                      nn.BatchNorm1d(h), nn.Dropout(cfg.dropout)]
            prev = h
        self.fusion = nn.Sequential(*layers)

        # Mean and log-variance outputs
        self.mean_head = nn.Linear(prev, 1)
        self.logvar_head = nn.Linear(prev, 1)

    def forward(
        self,
        x_M1: torch.Tensor,
        x_M2: torch.Tensor,
        x_M34_sym: torch.Tensor,
        g_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and logvar, each with shape (B,)."""
        h1 = self.enc_M1(x_M1)          # (B, 32)
        h2 = self.enc_M2(x_M2)          # (B, 32)
        h34 = self.enc_M34(x_M34_sym)   # (B, 32)

        z = torch.cat([h1, h2, h34, g_global], dim=1)  # (B, 3*32+52=148)
        h = self.fusion(z)

        mean = self.mean_head(h).squeeze(-1)
        logvar = torch.clamp(self.logvar_head(h).squeeze(-1), -10.0, 10.0)
        return mean, logvar


def gaussian_nll_loss(
    mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Gaussian NLL: 0.5 * [logvar + (y-mean)**2 / exp(logvar)]."""
    precision = torch.exp(-logvar)
    return 0.5 * (logvar + precision * (target - mean) ** 2).mean()


# Ensemble

@dataclass
class SymPrediction:
    """Prediction results from the symmetry-aware ensemble."""
    mean: np.ndarray                          # Ensemble mean, shape (N,)
    aleatoric: np.ndarray                     # Aleatoric variance, shape (N,)
    epistemic: np.ndarray                     # Epistemic variance, shape (N,)
    total_std: np.ndarray                     # Total predictive standard deviation, shape (N,)
    member_means: np.ndarray                  # (M, N)


class SymmetryAwareEnsembleV2:
    """Deep ensemble of SymmetryAwareMLP_V2 members.

    Inputs are x_M1, x_M2, x_M34_sym and g_global.
    """

    def __init__(self, config: SymmetryEnsembleConfigV2 | None = None):
        self.config = config or SymmetryEnsembleConfigV2()
        self.members: list[SymmetryAwareMLP_V2] = []
        # Fit normalization separately for each input block.
        self.stats_: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _standardize(self, data: dict[str, np.ndarray], fit: bool = False) -> dict[str, np.ndarray]:
        out = {}
        for key, arr in data.items():
            if fit:
                mu = arr.mean(axis=0)
                sd = arr.std(axis=0) + 1e-8
                self.stats_[key] = (mu, sd)
            mu, sd = self.stats_[key]
            out[key] = (arr - mu) / sd
        return out

    @staticmethod
    def _to_tensors(data: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
        return {k: torch.tensor(v, dtype=torch.float32, device=device)
                for k, v in data.items()}

    def fit(
        self,
        data: dict[str, np.ndarray],   # x_M1, x_M2, x_M34_sym, g_global
        y: np.ndarray,
        val_data: dict[str, np.ndarray] | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = False,
    ) -> "SymmetryAwareEnsembleV2":
        cfg = self.config
        device = torch.device(cfg.device)

        # Target normalization
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y) + 1e-8)
        ys = (y - self.y_mean_) / self.y_std_

        ds = self._standardize(data, fit=True)
        n = len(y)
        rng = np.random.default_rng(cfg.seed)

        has_val = val_data is not None and y_val is not None
        if has_val:
            dvs = self._standardize(val_data, fit=False)
            dvs_t = self._to_tensors(dvs, cfg.device)
            yv_t = torch.tensor((y_val - self.y_mean_) / self.y_std_,
                                dtype=torch.float32, device=device)

        self.members = []
        keys = ["x_M1", "x_M2", "x_M34_sym", "g_global"]

        for m in range(cfg.n_members):
            torch.manual_seed(cfg.seed + m)
            model = SymmetryAwareMLP_V2(cfg).to(device)

            idx = rng.integers(0, n, size=n) if cfg.bootstrap else np.arange(n)
            tensors = [torch.tensor(ds[k][idx], dtype=torch.float32) for k in keys]
            yt = torch.tensor(ys[idx], dtype=torch.float32)
            loader = DataLoader(TensorDataset(*tensors, yt),
                                batch_size=cfg.batch_size, shuffle=True)

            optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
            best_val, best_state, patience = float("inf"), None, 0

            for epoch in range(cfg.epochs):
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
            if verbose:
                print(f"  [Member {m+1}/{cfg.n_members}] complete"
                      + (f" val_NLL={best_val:.4f}" if has_val else ""))

        return self

    @torch.no_grad()
    def predict(self, data: dict[str, np.ndarray]) -> SymPrediction:
        cfg = self.config
        device = torch.device(cfg.device)
        ds = self._standardize(data, fit=False)
        keys = ["x_M1", "x_M2", "x_M34_sym", "g_global"]
        xt = [torch.tensor(ds[k], dtype=torch.float32, device=device) for k in keys]

        means, vars = [], []
        for model in self.members:
            model.eval()
            mean_s, logvar_s = model(*xt)
            # Restore the original target units.
            mean = mean_s.cpu().numpy() * self.y_std_ + self.y_mean_
            var = np.exp(logvar_s.cpu().numpy()) * (self.y_std_ ** 2)
            means.append(mean)
            vars.append(var)

        member_means = np.stack(means)          # (M, N)
        mean_ens = member_means.mean(axis=0)
        aleatoric = np.stack(vars).mean(axis=0)
        epistemic = member_means.var(axis=0)
        total = np.sqrt(aleatoric + epistemic)
        return SymPrediction(mean_ens, aleatoric, epistemic, total, member_means)

    def save(self, path: str | Path) -> None:
        torch.save({
            "config": self.config,
            "stats": self.stats_,
            "y_mean": self.y_mean_,
            "y_std": self.y_std_,
            "member_states": [m.state_dict() for m in self.members],
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "SymmetryAwareEnsembleV2":
        ckpt = torch.load(path, weights_only=False)
        obj = cls(ckpt["config"])
        obj.stats_ = ckpt["stats"]
        obj.y_mean_ = ckpt["y_mean"]
        obj.y_std_ = ckpt["y_std"]
        device = torch.device(obj.config.device)
        obj.members = []
        for state in ckpt["member_states"]:
            m = SymmetryAwareMLP_V2(obj.config).to(device)
            m.load_state_dict(state)
            m.eval()
            obj.members.append(m)
        return obj

