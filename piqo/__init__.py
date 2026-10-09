"""Piqo: Sovereign Entrypoint & Composition Root (PFD)."""

from piqo.engine.decision_engine import DecisionEngine, IDecisionEngine
from piqo.engine.layers.profiling_layer import ProfilingLayer
from piqo.engine.policies.slot_encoder import ILayoutEncoder, ITokenizer, SlotEncoder
from piqo.engine.policies.decision_resolver import DecisionResolver, IResolver
from piqo.engine.policies.torch_scorer import IScorer, TorchScorer
from piqo.engine.questions import (
    Answer,
    Choice,
    ChoiceResult,
    DecisionResult,
    DecisionSample,
    DecisionState,
    Noul,
    NoulResult,
    Question,
    QuestionSpec,
    Score,
    ScoreLayout,
    ScoreResult,
    TokenLayout,
    format_state,
)
from piqo.trainer.batches import SampleTargets, StepStats, TargetSlot, TrainingBatch
from piqo.trainer.layers.checkpointing_layer import CheckpointingLayer
from piqo.trainer.layers.metrics_layer import MetricsLayer
from piqo.trainer.policies.calibrated_loss import CalibratedLoss, ILoss
from piqo.trainer.policies.registry import (
    CHECKPOINTS_DIR,
    DEFAULT_CHECKPOINT,
    CheckpointInfo,
    get_latest_checkpoint,
    get_next_checkpoint_path,
    list_all_checkpoints,
)
from piqo.trainer.epoch_trainer import EpochTrainer, ITrainer

Trainer = EpochTrainer

__all__ = [
    # Core Primitives (P)
    "IDecisionEngine",
    "DecisionEngine",
    "ITrainer",
    "EpochTrainer",
    "Trainer",
    # Injected Opinion Sub-Primitives (Policies pi)
    "ILayoutEncoder",
    "SlotEncoder",
    "ITokenizer",
    "IScorer",
    "TorchScorer",
    "IResolver",
    "DecisionResolver",
    "ILoss",
    "CalibratedLoss",
    # Layered Co-Primitives (Lambda: P -> P)
    "ProfilingLayer",
    "CheckpointingLayer",
    "MetricsLayer",
    # Domain Models & DTOs
    "DecisionState",
    "Choice",
    "Noul",
    "Score",
    "Question",
    "ChoiceResult",
    "NoulResult",
    "ScoreResult",
    "Answer",
    "DecisionResult",
    "TokenLayout",
    "ScoreLayout",
    "TrainingBatch",
    "StepStats",
    "DecisionSample",
    "QuestionSpec",
    "CheckpointInfo",
    # Pure Colocated Transforms
    "format_state",
    # Model Registry
    "CHECKPOINTS_DIR",
    "DEFAULT_CHECKPOINT",
    "get_latest_checkpoint",
    "get_next_checkpoint_path",
    "list_all_checkpoints",
]
