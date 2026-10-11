"""Training Domain: Sovereign Optimization & Adaptation Primitive (PFD)."""

from training.stats import StepStats
from training.epoch_trainer import EpochTrainer, ITrainer
from training.layers.checkpointing_layer import CheckpointingLayer
from training.layers.metrics_layer import MetricsLayer
from training.checkpoints import (
    CHECKPOINTS_DIR,
    DEFAULT_CHECKPOINT,
    CheckpointInfo,
    CheckpointMeta,
    get_latest_checkpoint,
    get_next_checkpoint_path,
    list_all_checkpoints,
    resolve_default_checkpoint,
    save_checkpoint,
)

__all__ = [
    # Core Primitive
    "ITrainer",
    "EpochTrainer",
    # Layers
    "CheckpointingLayer",
    "MetricsLayer",
    # DTOs
    "StepStats",
    # Checkpoints
    "CHECKPOINTS_DIR",
    "DEFAULT_CHECKPOINT",
    "CheckpointInfo",
    "CheckpointMeta",
    "get_latest_checkpoint",
    "get_next_checkpoint_path",
    "list_all_checkpoints",
    "resolve_default_checkpoint",
    "save_checkpoint",
]
