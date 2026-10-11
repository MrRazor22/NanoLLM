"""Training Domain: Sovereign Optimization & Adaptation Primitive (PFD)."""

from training.batches import SampleTargets, StepStats, TargetSlot, TrainingBatch
from training.epoch_trainer import EpochTrainer, ITrainer
from training.layers.checkpointing_layer import CheckpointingLayer
from training.layers.metrics_layer import MetricsLayer
from training.policies.batch_collator import IBatchCollator, PaddedBatchCollator
from training.policies.calibrated_loss import CalibratedLoss, ILoss
from training.checkpoints import (
    CHECKPOINTS_DIR,
    DEFAULT_CHECKPOINT,
    CheckpointInfo,
    get_latest_checkpoint,
    get_next_checkpoint_path,
    list_all_checkpoints,
    resolve_default_checkpoint,
)

__all__ = [
    # Core Primitive
    "ITrainer",
    "EpochTrainer",
    # Policies
    "ILoss",
    "CalibratedLoss",
    "IBatchCollator",
    "PaddedBatchCollator",
    # Layers
    "CheckpointingLayer",
    "MetricsLayer",
    # DTOs
    "SampleTargets",
    "StepStats",
    "TargetSlot",
    "TrainingBatch",
    # Checkpoints
    "CHECKPOINTS_DIR",
    "DEFAULT_CHECKPOINT",
    "CheckpointInfo",
    "get_latest_checkpoint",
    "get_next_checkpoint_path",
    "list_all_checkpoints",
    "resolve_default_checkpoint",
]
