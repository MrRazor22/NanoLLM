"""Piqo: Sovereign Composition Root (PFD)."""

from model import (
    IModel, TorchModel,
    TokenLayout, ScoreLayout,
    ILoss, CalibratedLoss,
)
from inference import (
    IDecisionEngine, DecisionEngine,
    ILayoutEncoder, SlotEncoder, ITokenizer, IResolver, DecisionResolver,
    ProfilingLayer,
    DecisionState, Choice, Noul, Score, Question,
    IAnswer, ChoiceResult, NoulResult, ScoreResult, Answer, DecisionResult,
    DecisionCase, format_state,
)
from training import (
    ITrainer, EpochTrainer,
    IBatchCollator, PaddedBatchCollator,
    CheckpointingLayer, MetricsLayer,
    Batch, TrainingBatch, SampleTargets, StepStats, TargetSlot,
    CHECKPOINTS_DIR, DEFAULT_CHECKPOINT, CheckpointInfo, CheckpointMeta,
    get_latest_checkpoint, get_next_checkpoint_path, list_all_checkpoints, resolve_default_checkpoint, save_checkpoint,
)
from data import (
    IDataset, Dataset, DataSplits, SourceRecord,
    ICachePolicy, DiskCachePolicy, ISplitPolicy, BudgetSplitPolicy, ILeakagePolicy, NGramLeakagePolicy,
    IDataSource, GlaiveSource, DecisionsSource, GenericSource,
    DATA_DIR, CORPUS_DIR, RAW_DIR, ADAPTED_DIR,
)
from evaluation import (
    IScorecard, Scorecard, BenchmarkItem, BenchmarkReport, BaselineTrack, CategoryHits,
)

__all__ = [
    # Core Primitives
    "IModel", "TorchModel",
    "IDecisionEngine", "DecisionEngine",
    "ITrainer", "EpochTrainer",
    "IDataset", "Dataset",
    "IScorecard", "Scorecard",
    # Policies & Injected Sub-Primitives
    "ILoss", "CalibratedLoss",
    "ILayoutEncoder", "SlotEncoder", "ITokenizer", "IResolver", "DecisionResolver",
    "IBatchCollator", "PaddedBatchCollator",
    "ISplitPolicy", "BudgetSplitPolicy", "ILeakagePolicy", "NGramLeakagePolicy", "ICachePolicy", "DiskCachePolicy",
    "IDataSource", "GlaiveSource", "DecisionsSource", "GenericSource",
    # Layers (Lambda: P -> P)
    "ProfilingLayer", "CheckpointingLayer", "MetricsLayer",
    # Domain DTOs & Models
    "DecisionState", "Choice", "Noul", "Score", "Question",
    "IAnswer", "ChoiceResult", "NoulResult", "ScoreResult", "Answer", "DecisionResult",
    "TokenLayout", "ScoreLayout", "DecisionCase",
    "Batch", "TrainingBatch", "StepStats", "TargetSlot", "SampleTargets",
    "CheckpointInfo", "CheckpointMeta", "BenchmarkItem", "BenchmarkReport", "BaselineTrack", "CategoryHits",
    # Transforms & Paths
    "format_state", "CHECKPOINTS_DIR", "DEFAULT_CHECKPOINT",
    "DATA_DIR", "CORPUS_DIR", "RAW_DIR", "ADAPTED_DIR",
    "get_latest_checkpoint", "get_next_checkpoint_path", "list_all_checkpoints", "resolve_default_checkpoint",
    "save_checkpoint",
]
