"""Deep-ensemble regressor with independently initialized heteroscedastic MLPs.

The predictive mean averages member means. Aleatoric variance averages
member variances; epistemic variance is the variance of member means.
Total standard deviation is sqrt(aleatoric variance + epistemic variance).
The ensemble supplies uncertainty for active-learning acquisition.

Reference: Lakshminarayanan et al., NeurIPS 2017.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .mlp import HeteroscedasticMLP, gaussian_nll_loss


@dataclass
class EnsembleConfig:
    """Deep-ensemble configuration."""

    input_dim: int = 77
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.1
    n_members: int = 10
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 30
    bootstrap: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PredictionResult:
    """Container for ensemble prediction results."""

    mean: np.ndarray  # Ensemble predictive mean
    epistemic: np.ndarray  # Epistemic standard deviation across member means
    aleatoric: np.ndarray  # Square root of the mean predicted member variance
    total_std: np.ndarray  # Total predictive standard deviation
    member_means: np.ndarray = field(repr=False)  # Member predictions, shape (n_members,N)


class DeepEnsemble:
    """Deep-ensemble regressor.

    Example:
        cfg = EnsembleConfig(n_members=10)
        ens = DeepEnsemble(cfg)
        ens.fit(X_train, y_train, X_val, y_val)
        result = ens.predict(X_test)
        # Access result.mean, result.total_std, etc.
    """

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        self.members: list[HeteroscedasticMLP] = []
        # Normalization parameters fitted during training
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self._fitted = False

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def _standardize_x(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean_) / self.x_std_

    def _standardize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean_) / self.y_std_

    def _destandardize_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std_ + self.y_mean_

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> "DeepEnsemble":
        """Train all ensemble members."""
        cfg = self.config
        device = torch.device(cfg.device)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Fit normalization parameters.
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-8
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std() + 1e-8)

        Xs = self._standardize_x(X)
        ys = self._standardize_y(y)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xvs = self._standardize_x(np.asarray(X_val, dtype=np.float64))
            yvs = self._standardize_y(np.asarray(y_val, dtype=np.float64).ravel())
            Xv_t = torch.tensor(Xvs, dtype=torch.float32, device=device)
            yv_t = torch.tensor(yvs, dtype=torch.float32, device=device)

        self.members = []
        n = len(Xs)
        rng = np.random.default_rng(cfg.seed)

        for m in range(cfg.n_members):
            # Use different member seeds for initialization and bootstrap sampling.
            torch.manual_seed(cfg.seed + m)
            model = HeteroscedasticMLP(
                input_dim=cfg.input_dim,
                hidden_dims=cfg.hidden_dims,
                dropout=cfg.dropout,
            ).to(device)

            # Bootstrap resampling
            if cfg.bootstrap:
                idx = rng.integers(0, n, size=n)
            else:
                idx = np.arange(n)

            Xm = torch.tensor(Xs[idx], dtype=torch.float32, device=device)
            ym = torch.tensor(ys[idx], dtype=torch.float32, device=device)
            loader = DataLoader(
                TensorDataset(Xm, ym),
                batch_size=cfg.batch_size,
                shuffle=True,
            )

            optim = torch.optim.Adam(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )

            best_val = float("inf")
            best_state = None
            patience = 0

            for epoch in range(cfg.epochs):
                model.train()
                for xb, yb in loader:
                    optim.zero_grad()
                    mean, logvar = model(xb)
                    loss = gaussian_nll_loss(mean, logvar, yb)
                    loss.backward()
                    optim.step()

                if has_val:
                    model.eval()
                    with torch.no_grad():
                        vm, vlog = model(Xv_t)
                        vloss = gaussian_nll_loss(vm, vlog, yv_t).item()
                    if vloss < best_val - 1e-5:
                        best_val = vloss
                        best_state = {
                            k: v.detach().clone() for k, v in model.state_dict().items()
                        }
                        patience = 0
                    else:
                        patience += 1
                        if patience >= cfg.early_stop_patience:
                            break

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.members.append(model)

            if verbose:
                tag = f"val_nll={best_val:.4f}" if has_val else "no_val"
                print(f"  [Member {m+1}/{cfg.n_members}] training complete ({tag})")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 8192) -> PredictionResult:
        """Predict the mean and uncertainty components."""
        if not self._fitted:
            raise RuntimeError("Model is not trained; call fit() first")

        device = torch.device(self.config.device)
        Xs = self._standardize_x(np.asarray(X, dtype=np.float64))
        Xt = torch.tensor(Xs, dtype=torch.float32, device=device)

        member_means = []  # Standardized predictions, shape (n_members,N)
        member_vars = []  # Standardized predictions, shape (n_members,N)

        for model in self.members:
            model.eval()
            means_chunks = []
            vars_chunks = []
            for i in range(0, len(Xt), batch_size):
                xb = Xt[i : i + batch_size]
                mean, logvar = model(xb)
                means_chunks.append(mean.cpu().numpy())
                vars_chunks.append(torch.exp(logvar).cpu().numpy())
            member_means.append(np.concatenate(means_chunks))
            member_vars.append(np.concatenate(vars_chunks))

        member_means = np.array(member_means)  # (M, N)
        member_vars = np.array(member_vars)  # (M, N)

        # Ensemble statistics in standardized units
        ens_mean = member_means.mean(axis=0)
        epistemic_var = member_means.var(axis=0)  # Variance of member means
        aleatoric_var = member_vars.mean(axis=0)  # Mean of member variances
        total_var = epistemic_var + aleatoric_var

        # Restore target units; variances scale by y_std**2.
        y_std2 = self.y_std_**2
        return PredictionResult(
            mean=self._destandardize_y(ens_mean),
            epistemic=np.sqrt(epistemic_var * y_std2),
            aleatoric=np.sqrt(aleatoric_var * y_std2),
            total_std=np.sqrt(total_var * y_std2),
            member_means=self._destandardize_y(member_means),
        )

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.__dict__,
                "x_mean": self.x_mean_,
                "x_std": self.x_std_,
                "y_mean": self.y_mean_,
                "y_std": self.y_std_,
                "members": [m.state_dict() for m in self.members],
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DeepEnsemble":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = EnsembleConfig(**ckpt["config"])
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        obj = cls(cfg)
        obj.x_mean_ = ckpt["x_mean"]
        obj.x_std_ = ckpt["x_std"]
        obj.y_mean_ = ckpt["y_mean"]
        obj.y_std_ = ckpt["y_std"]
        device = torch.device(cfg.device)
        obj.members = []
        for state in ckpt["members"]:
            model = HeteroscedasticMLP(
                input_dim=cfg.input_dim,
                hidden_dims=cfg.hidden_dims,
                dropout=cfg.dropout,
            ).to(device)
            model.load_state_dict(state)
            model.eval()
            obj.members.append(model)
        obj._fitted = True
        return obj
