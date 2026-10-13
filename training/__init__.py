"""Training Domain: Sovereign Optimization & Adaptation Primitive (PFD)."""

from training.stats import StepStats
from training.epoch_trainer import EpochTrainer, ITrainer
from training.telemetry import CheckpointingLayer, MetricsLayer

__all__ = [
    # Core Primitive
    "ITrainer",
    "EpochTrainer",
    # Layers
    "CheckpointingLayer",
    "MetricsLayer",
    # DTOs
    "StepStats",
]
