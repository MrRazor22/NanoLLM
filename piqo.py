"""Piqo: Sovereign Composition Root (PFD)."""

from model import (
    IModel, TorchModel,
    TokenLayout, PredictionLayout, Batch, SampleTargets, TargetSlot,
    ILoss, CalibratedLoss,
    CHECKPOINTS_DIR, DEFAULT_CHECKPOINT, CheckpointInfo, CheckpointMeta,
    get_latest_checkpoint, get_next_checkpoint_path, list_all_checkpoints, resolve_default_checkpoint,
    resolve_device,
)
from inference import (
    IDecisionEngine, DecisionEngine,
    IResolver, DecisionResolver,
    ProfilingLayer,
    DecisionState, Choice, Noul, Score, Question,
    IAnswer, ChoiceResult, NoulResult, ScoreResult, Answer, DecisionResult,
    format_state,
)
from training import (
    ITrainer, EpochTrainer,
    CheckpointingLayer, MetricsLayer,
    StepStats,
)
from data import (
    IDataset, Dataset, DataSplits, SourceRecord, DecisionCase,
    ICachePolicy, DiskCachePolicy, ISplitPolicy, BudgetSplitPolicy, ILeakagePolicy, NGramLeakagePolicy,
    IDataSource, GlaiveSource, DecisionsSource, HuggingFaceSource, BenchmarkSpec, standard_sources,
    DATA_DIR, CORPUS_DIR, RAW_DIR,
)
from benchmark import (
    IScorecard, Scorecard, BenchmarkItem, BenchmarkReport, SCORES_FILE, CategoryHits,
    ILedgerPolicy, DiskLedgerPolicy,
    GlinerDecideEngine, LayaEngine, VerdictEngine,
)

__all__ = [
    # Core Primitives
    "IModel", "TorchModel",
    "IDecisionEngine", "DecisionEngine",
    "ITrainer", "EpochTrainer",
    "IDataset", "Dataset",
    "IScorecard", "Scorecard", "GlinerDecideEngine", "LayaEngine", "VerdictEngine",
    # Policies & Injected Sub-Primitives
    "ILoss", "CalibratedLoss",
    "IResolver", "DecisionResolver",
    "ISplitPolicy", "BudgetSplitPolicy", "ILeakagePolicy", "NGramLeakagePolicy", "ICachePolicy", "DiskCachePolicy",
    "ILedgerPolicy", "DiskLedgerPolicy",
    "IDataSource", "GlaiveSource", "DecisionsSource", "HuggingFaceSource", "BenchmarkSpec", "standard_sources",
    # Layers (Lambda: P -> P)
    "ProfilingLayer", "CheckpointingLayer", "MetricsLayer",
    # Domain DTOs & Models
    "DecisionState", "Choice", "Noul", "Score", "Question",
    "IAnswer", "ChoiceResult", "NoulResult", "ScoreResult", "Answer", "DecisionResult",
    "TokenLayout", "PredictionLayout", "DecisionCase",
    "Batch", "StepStats", "TargetSlot", "SampleTargets",
    "CheckpointInfo", "CheckpointMeta", "BenchmarkItem", "BenchmarkReport", "SCORES_FILE", "CategoryHits",
    # Transforms & Paths
    "format_state", "CHECKPOINTS_DIR", "DEFAULT_CHECKPOINT",
    "DATA_DIR", "CORPUS_DIR", "RAW_DIR",
    "get_latest_checkpoint", "get_next_checkpoint_path", "list_all_checkpoints", "resolve_default_checkpoint",
    "resolve_device",
]
