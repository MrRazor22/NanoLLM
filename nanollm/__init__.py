"""NanoLLM: Sovereign Entrypoint & Composition Root (PFD)."""

from nanollm.engine.decision_engine import DecisionEngine, IDecisionEngine
from nanollm.engine.layers.profiling_layer import ProfilingLayer
from nanollm.engine.policies.slot_encoder import ILayoutEncoder, ITokenizer, SlotEncoder
from nanollm.engine.policies.decision_resolver import DecisionResolver, IResolver
from nanollm.engine.policies.torch_scorer import IScorer, TorchScorer
from nanollm.engine.questions import (
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
from nanollm.trainer.batches import SampleTargets, StepStats, TargetSlot, TrainingBatch
from nanollm.trainer.layers.checkpointing_layer import CheckpointingLayer
from nanollm.trainer.layers.metrics_layer import MetricsLayer
from nanollm.trainer.policies.calibrated_loss import CalibratedLoss, ILoss
from nanollm.trainer.policies.registry import (
    CHECKPOINTS_DIR,
    DEFAULT_CHECKPOINT,
    CheckpointInfo,
    get_latest_checkpoint,
    get_next_checkpoint_path,
    list_all_checkpoints,
)
from nanollm.trainer.epoch_trainer import EpochTrainer, ITrainer

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
