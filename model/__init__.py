"""Model Domain: Sovereign Neural Computation Primitive (PFD)."""

from model.calibrated_loss import CalibratedLoss, ILoss
from model.layouts import (
    CHECKPOINTS_DIR,
    DEFAULT_CHECKPOINT,
    Batch,
    CheckpointInfo,
    CheckpointMeta,
    PredictionLayout,
    SampleTargets,
    TargetSlot,
    TokenLayout,
    collate_batch,
    encode_layout,
    get_latest_checkpoint,
    get_next_checkpoint_path,
    list_all_checkpoints,
    resolve_default_checkpoint,
    resolve_device,
)
from model.torch_model import IModel, TorchModel

__all__ = [
    # Primitive Contracts & Implementations
    "IModel",
    "TorchModel",
    # Policies
    "ILoss",
    "CalibratedLoss",
    # Checkpoints & Weights
    "CHECKPOINTS_DIR",
    "DEFAULT_CHECKPOINT",
    "CheckpointInfo",
    "CheckpointMeta",
    "get_latest_checkpoint",
    "get_next_checkpoint_path",
    "list_all_checkpoints",
    "resolve_default_checkpoint",
    # Domain Extensions
    "resolve_device",
    # Model Layout DTOs & Transforms
    "TokenLayout",
    "PredictionLayout",
    "TargetSlot",
    "SampleTargets",
    "Batch",
    "encode_layout",
    "collate_batch",
]
