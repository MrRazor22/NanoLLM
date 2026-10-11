"""Piqo: Sovereign Composition Root (PFD)."""

from model import (
    IModel, TorchModel,
    TokenLayout, ScoreLayout, Batch, SampleTargets, TargetSlot,
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
    StepStats,
    CHECKPOINTS_DIR, DEFAULT_CHECKPOINT, CheckpointInfo, CheckpointMeta,
    get_latest_checkpoint, get_next_checkpoint_path, list_all_checkpoints, resolve_default_checkpoint, save_checkpoint,
)
from data import (
    IDataset, Dataset, DataSplits, SourceRecord,
    ICachePolicy, DiskCachePolicy, ISplitPolicy, BudgetSplitPolicy, ILeakagePolicy, NGramLeakagePolicy,
    IDataSource, GlaiveSource, DecisionsSource, HuggingFaceSource, standard_sources,
    DATA_DIR, CORPUS_DIR, RAW_DIR, ADAPTED_DIR,
)
from benchmark import (
    IScorecard, Scorecard, BenchmarkItem, BenchmarkReport, SCORES_FILE, CategoryHits,
    ILedgerPolicy, DiskLedgerPolicy,
    LayaEngine, VerdictEngine,
)

__all__ = [
    # Core Primitives
    "IModel", "TorchModel",
    "IDecisionEngine", "DecisionEngine",
    "ITrainer", "EpochTrainer",
    "IDataset", "Dataset",
    "IScorecard", "Scorecard", "LayaEngine", "VerdictEngine",
    # Policies & Injected Sub-Primitives
    "ILoss", "CalibratedLoss",
    "ILayoutEncoder", "SlotEncoder", "ITokenizer", "IResolver", "DecisionResolver",
    "IBatchCollator", "PaddedBatchCollator",
    "ISplitPolicy", "BudgetSplitPolicy", "ILeakagePolicy", "NGramLeakagePolicy", "ICachePolicy", "DiskCachePolicy",
    "ILedgerPolicy", "DiskLedgerPolicy",
    "IDataSource", "GlaiveSource", "DecisionsSource", "HuggingFaceSource", "standard_sources",
    # Layers (Lambda: P -> P)
    "ProfilingLayer", "CheckpointingLayer", "MetricsLayer",
    # Domain DTOs & Models
    "DecisionState", "Choice", "Noul", "Score", "Question",
    "IAnswer", "ChoiceResult", "NoulResult", "ScoreResult", "Answer", "DecisionResult",
    "TokenLayout", "ScoreLayout", "DecisionCase",
    "Batch", "StepStats", "TargetSlot", "SampleTargets",
    "CheckpointInfo", "CheckpointMeta", "BenchmarkItem", "BenchmarkReport", "SCORES_FILE", "CategoryHits",
    # Transforms & Paths
    "format_state", "CHECKPOINTS_DIR", "DEFAULT_CHECKPOINT",
    "DATA_DIR", "CORPUS_DIR", "RAW_DIR", "ADAPTED_DIR",
    "get_latest_checkpoint", "get_next_checkpoint_path", "list_all_checkpoints", "resolve_default_checkpoint",
    "save_checkpoint",
]
