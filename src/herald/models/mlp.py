"""Heteroscedastic MLP regressor used as a deep-ensemble member.

Each network predicts a mean and log variance. Gaussian negative
log-likelihood training estimates aleatoric uncertainty.

Reference: Lakshminarayanan et al., Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles, NeurIPS 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HeteroscedasticMLP(nn.Module):
    """MLP predicting a mean and log variance.

    Architecture: input -> ReLU/Dropout hidden layers -> mean/logvar heads.

    Args:
        input_dim: Input-feature dimension, 77 in the original feature set.
        hidden_dims: Hidden-layer widths.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 77,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)

        # Separate mean and log-variance outputs
        self.mean_head = nn.Linear(prev, 1)
        self.logvar_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and logvar, each with shape (N,)."""
        feat = self.backbone(x)
        mean = self.mean_head(feat).squeeze(-1)
        logvar = self.logvar_head(feat).squeeze(-1)
        # Bound log variance for numerical stability.
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mean, logvar


def gaussian_nll_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss.

    NLL = 0.5 * [logvar + (target-mean)**2 / exp(logvar)].

    Args:
        mean: Predicted means, shape (N,).
        logvar: Predicted log variances, shape (N,).
        target: Reference values, shape (N,).

    Returns:
        Scalar loss averaged over the batch.
    """
    precision = torch.exp(-logvar)
    return 0.5 * (logvar + precision * (target - mean) ** 2).mean()
