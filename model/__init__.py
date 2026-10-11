"""Model Domain: Sovereign Neural Computation Primitive (PFD)."""

from model.layouts import (
    Batch,
    ScoreLayout,
    SampleTargets,
    TargetSlot,
    TokenLayout,
    TrainingBatch,
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
    # Model Layout DTOs
    "TokenLayout",
    "ScoreLayout",
    "TargetSlot",
    "SampleTargets",
    "Batch",
    "TrainingBatch",
]
