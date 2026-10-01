"""Deep Ensemble models."""
from herald.models.mlp import HeteroscedasticMLP, gaussian_nll_loss
from herald.models.deep_ensemble import DeepEnsemble, EnsembleConfig, PredictionResult

__all__ = [
    "HeteroscedasticMLP",
    "gaussian_nll_loss",
    "DeepEnsemble",
    "EnsembleConfig",
    "PredictionResult",
]
