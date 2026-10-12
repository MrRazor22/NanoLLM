"""Model Domain: Sovereign Neural Computation Primitive (PFD)."""

from model.device import resolve_device
from model.layouts import (
    Batch,
    PredictionLayout,
    ScoreLayout,
    SampleTargets,
    TargetSlot,
    TokenLayout,
)
from model.torch_model import IModel, TorchModel
from model.policies.calibrated_loss import CalibratedLoss, ILoss

__all__ = [
    # Primitive Contracts & Implementations
    "IModel",
    "TorchModel",
    # Policies
    "ILoss",
    "CalibratedLoss",
    # Device Resolution
    "resolve_device",
    # Model Layout DTOs
    "TokenLayout",
    "ScoreLayout",
    "PredictionLayout",
    "TargetSlot",
    "SampleTargets",
    "Batch",
]
