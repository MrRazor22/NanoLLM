"""Piqo: Sovereign Composition Root (PFD)."""

from inference import (
    IDecisionEngine, DecisionEngine,
    ILayoutEncoder, SlotEncoder, ITokenizer, IScorer, TorchScorer, IResolver, DecisionResolver,
    ProfilingLayer,
    DecisionState, Choice, Noul, Score, Question,
    IAnswer, ChoiceResult, NoulResult, ScoreResult, Answer, DecisionResult,
    TokenLayout, ScoreLayout, DecisionCase, format_state,
)
from training import (
    ITrainer, EpochTrainer,
    ILoss, CalibratedLoss, IBatchCollator, PaddedBatchCollator,
    CheckpointingLayer, MetricsLayer,
    SampleTargets, StepStats, TargetSlot, TrainingBatch,
    CHECKPOINTS_DIR, DEFAULT_CHECKPOINT, CheckpointInfo,
    get_latest_checkpoint, get_next_checkpoint_path, list_all_checkpoints, resolve_default_checkpoint,
)
from data import (
    IDataset, Dataset, DataSplits, SourceRecord,
    ICachePolicy, DiskCachePolicy, ISplitPolicy, BudgetSplitPolicy, ILeakagePolicy, NGramLeakagePolicy,
    IDataSource, GlaiveSource, DecisionsSource, GenericSource,
    DATA_DIR, CORPUS_DIR, RAW_DIR, ADAPTED_DIR,
)
from evaluation import (
    IEvaluator, Scorecard, BenchmarkItem, BenchmarkReport,
)

__all__ = [
    # Core Primitives
    "IDecisionEngine", "DecisionEngine", "ITrainer", "EpochTrainer",
    "IDataset", "Dataset", "IEvaluator", "Scorecard",
    # Policies & Injected Sub-Primitives
    "ILayoutEncoder", "SlotEncoder", "ITokenizer", "IScorer", "TorchScorer", "IResolver", "DecisionResolver",
    "ILoss", "CalibratedLoss", "IBatchCollator", "PaddedBatchCollator",
    "ISplitPolicy", "BudgetSplitPolicy", "ILeakagePolicy", "NGramLeakagePolicy", "ICachePolicy", "DiskCachePolicy",
    "IDataSource", "GlaiveSource", "DecisionsSource", "GenericSource",
    # Layers (Lambda: P -> P)
    "ProfilingLayer", "CheckpointingLayer", "MetricsLayer",
    # Domain DTOs & Models
    "DecisionState", "Choice", "Noul", "Score", "Question",
    "IAnswer", "ChoiceResult", "NoulResult", "ScoreResult", "Answer", "DecisionResult",
    "TokenLayout", "ScoreLayout", "DecisionCase", "TrainingBatch", "StepStats", "TargetSlot", "SampleTargets",
    "CheckpointInfo", "BenchmarkItem", "BenchmarkReport",
    # Transforms & Paths
    "format_state", "CHECKPOINTS_DIR", "DEFAULT_CHECKPOINT",
    "DATA_DIR", "CORPUS_DIR", "RAW_DIR", "ADAPTED_DIR",
    "get_latest_checkpoint", "get_next_checkpoint_path", "list_all_checkpoints", "resolve_default_checkpoint",
]
