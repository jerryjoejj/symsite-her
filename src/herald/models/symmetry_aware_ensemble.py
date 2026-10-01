"""Symmetry-aware heteroscedastic ensemble with shared remote-site encoding.

M1/M2 have independent encoders from 13 to 32 dimensions. M3/M4 share
an encoder with the same dimensions, followed by elementwise MaxPool.
The fusion network uses widths 148 -> 128 -> 64 and mean/log-variance
outputs trained with Gaussian NLL. Shared encoding and symmetric pooling
make predictions invariant to M3/M4 exchange at numerical precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Configuration

@dataclass
class SymmetryEnsembleConfig:
    """Symmetry-aware ensemble configuration."""

    site_dim: int = 13          # Raw elemental-property dimension per site (PROP13)
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
    """Encode one site from site_dim to site_embed dimensions."""

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


class SymmetryAwareMLP(nn.Module):
    """One symmetry-aware heteroscedastic MLP member.

    Shared remote encoding and MaxPool enforce M3/M4 exchange invariance.
    Independent encoders retain M1/M2 roles.
    """

    def __init__(self, cfg: SymmetryEnsembleConfig):
        super().__init__()
        self.cfg = cfg
        d, e = cfg.site_dim, cfg.site_embed

        # Independent encoders for the adsorption center M1 and neighbor M2
        self.enc_M1 = SiteEncoder(d, e, cfg.dropout)
        self.enc_M2 = SiteEncoder(d, e, cfg.dropout)
        # Shared encoder parameters for M3/M4
        self.enc_shared = SiteEncoder(d, e, cfg.dropout)

        # Fusion network: 3*embed + global_dim -> fusion_dims
        fusion_in = 3 * e + cfg.global_dim
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
        x_M1: torch.Tensor, x_M2: torch.Tensor,
        x_M3: torch.Tensor, x_M4: torch.Tensor,
        g_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and logvar, each with shape (B,)."""
        h1 = self.enc_M1(x_M1)          # (B, e)
        h2 = self.enc_M2(x_M2)          # (B, e)
        h3 = self.enc_shared(x_M3)      # Shape (B,e), with shared parameters
        h4 = self.enc_shared(x_M4)      # Shape (B,e), with shared parameters

        # Elementwise MaxPool makes remote-site pooling exchange-invariant.
        h_sym = torch.maximum(h3, h4)   # (B, e)

        z = torch.cat([h1, h2, h_sym, g_global], dim=1)  # (B, 3e+52)
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
    member_means: np.ndarray = field(repr=False)  # (M, N)


class SymmetryAwareEnsemble:
    """Deep ensemble of SymmetryAwareMLP members.

    Inputs are x_M1, x_M2, x_M3, x_M4, each (N,13), and g_global (N,52).
    The ensemble fits and applies normalization for each input block.
    """

    def __init__(self, config: SymmetryEnsembleConfig | None = None):
        self.config = config or SymmetryEnsembleConfig()
        self.members: list[SymmetryAwareMLP] = []
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
        data: dict[str, np.ndarray],   # x_M1..x_M4, g_global
        y: np.ndarray,
        val_data: dict[str, np.ndarray] | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = False,
    ) -> "SymmetryAwareEnsemble":
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
        keys = ["x_M1", "x_M2", "x_M3", "x_M4", "g_global"]

        for m in range(cfg.n_members):
            torch.manual_seed(cfg.seed + m)
            model = SymmetryAwareMLP(cfg).to(device)

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
        keys = ["x_M1", "x_M2", "x_M3", "x_M4", "g_global"]
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
    def load(cls, path: str | Path) -> "SymmetryAwareEnsemble":
        ckpt = torch.load(path, weights_only=False)
        obj = cls(ckpt["config"])
        obj.stats_ = ckpt["stats"]
        obj.y_mean_ = ckpt["y_mean"]
        obj.y_std_ = ckpt["y_std"]
        device = torch.device(obj.config.device)
        obj.members = []
        for state in ckpt["member_states"]:
            m = SymmetryAwareMLP(obj.config).to(device)
            m.load_state_dict(state)
            m.eval()
            obj.members.append(m)
        return obj
