"""Piqo: Sovereign Composition Root (PFD)."""

from model.torch_model import IModel, TorchModel
from model.calibrated_loss import ILoss, CalibratedLoss
from model.layouts.token_layout import TokenLayout, PredictionLayout, Batch, SampleTargets, TargetSlot
from model.layouts.checkpoints import (
    CHECKPOINTS_DIR, DEFAULT_CHECKPOINT, CheckpointInfo, CheckpointMeta,
    get_latest_checkpoint, get_next_checkpoint_path, list_all_checkpoints, resolve_default_checkpoint,
    resolve_device,
)
from inference.decision_engine import IDecisionEngine, DecisionEngine
from inference.decision_resolver import IResolver, DecisionResolver
from inference.profiling_layer import ProfilingLayer
from inference.questions import (
    DecisionState, Choice, Noul, Score, Question,
    IAnswer, ChoiceResult, NoulResult, ScoreResult, Answer, DecisionResult,
    format_state,
)
from training.epoch_trainer import ITrainer, EpochTrainer
from training.policies.schedule_policy import ISchedulePolicy, CosineSchedulePolicy, ConstantSchedulePolicy
from training.telemetry.checkpointing_layer import CheckpointingLayer
from training.telemetry.metrics_layer import MetricsLayer
from training.stats import StepStats
from data.dataset import IDataset, Dataset, DATA_DIR, CORPUS_DIR
from data.case import DataSplits, SourceRecord, DecisionCase, IDataSource
from data.disk_cache import ICachePolicy, DiskCachePolicy
from data.curation.budget_split import ISplitPolicy, BudgetSplitPolicy
from data.curation.ngram_leakage import ILeakagePolicy, NGramLeakagePolicy
from data.sources.glaive_source import GlaiveSource
from data.sources.decisions_source import DecisionsSource
from data.sources.huggingface_source import HuggingFaceSource, BenchmarkSpec, RAW_DIR
from benchmark.scorecard import IScorecard, Scorecard, BenchmarkItem, BenchmarkReport, SCORES_FILE, CategoryHits
from benchmark.ledger import ILedgerPolicy, DiskLedgerPolicy
from benchmark.competitors.gliner_decide import GlinerDecideEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine

__all__ = [
    # Core Primitives
    "IModel", "TorchModel",
    "IDecisionEngine", "DecisionEngine",
    "ITrainer", "EpochTrainer",
    "IDataset", "Dataset",
    "IScorecard", "Scorecard", "GlinerDecideEngine", "LayaEngine", "VerdictEngine",
    # Policies & Injected Sub-Primitives
    "ILoss", "CalibratedLoss",
    "ISchedulePolicy", "CosineSchedulePolicy", "ConstantSchedulePolicy",
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
