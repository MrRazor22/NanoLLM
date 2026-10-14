"""Piqo: Sovereign Composition Root (PFD)."""

from model.torch_model import IModel, TorchModel
from model.policies.calibrated_loss import ILoss, CalibratedLoss
from model.policies.layout_policy import ILayoutPolicy, MaskedLayoutPolicy
from model.policies.weight_cache import (
    DEFAULT_WEIGHTS_DIR, DiskWeightCache, IWeightCache,
)
from model.model_extensions import list_weights, resolve_device
from model.weights import Weights
from model.token_layout import TokenLayout, PredictionLayout, Batch, SampleTargets, TargetSlot
from inference.decision_engine import IDecisionEngine, DecisionEngine
from inference.decision_resolver import IResolver, DecisionResolver
from inference.profiling_layer import ProfilingLayer
from inference.questions import (
    DecisionState, Choice, Noul, Score, Question,
    IAnswer, ChoiceResult, NoulResult, ScoreResult, Answer, DecisionResult,
    format_state,
)
from training.epoch_trainer import ITrainer, EpochTrainer
from training.schedule_policy import ISchedulePolicy, CosineSchedulePolicy, ConstantSchedulePolicy
from training.telemetry.checkpointing_layer import CheckpointingLayer
from training.telemetry.metrics_layer import MetricsLayer
from training.stats import StepStats
from data.dataset import IDataset, Dataset, DATA_DIR, CORPUS_DIR, standard_sources
from data.case import DataSplits, SourceRecord, DecisionCase, IDataSource
from data.disk_cache import ICachePolicy, DiskCachePolicy
from data.curation.budget_split import ISplitPolicy, BudgetSplitPolicy
from data.curation.ngram_leakage import ILeakagePolicy, NGramLeakagePolicy
from data.sources.glaive_source import GlaiveSource
from data.sources.decisions_source import DecisionsSource
from data.sources.huggingface_source import HuggingFaceSource, BenchmarkSpec, RAW_DIR
from benchmark.scorecard import IScorecard, Scorecard, BenchmarkReport, CategoryHits
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
    "ILayoutPolicy", "MaskedLayoutPolicy",
    "IWeightCache", "DiskWeightCache", "DEFAULT_WEIGHTS_DIR",
    "ISchedulePolicy", "CosineSchedulePolicy", "ConstantSchedulePolicy",
    "IResolver", "DecisionResolver",
    "ISplitPolicy", "BudgetSplitPolicy", "ILeakagePolicy", "NGramLeakagePolicy", "ICachePolicy", "DiskCachePolicy",
    "ILedgerPolicy", "DiskLedgerPolicy",
    "IDataSource", "GlaiveSource", "DecisionsSource", "HuggingFaceSource", "BenchmarkSpec", "standard_sources",
    # Layers (Lambda: P -> P)
    "ProfilingLayer", "CheckpointingLayer", "MetricsLayer",
    # Domain DTOs & Models
    "Weights",
    "DecisionState", "Choice", "Noul", "Score", "Question",
    "IAnswer", "ChoiceResult", "NoulResult", "ScoreResult", "Answer", "DecisionResult",
    "TokenLayout", "PredictionLayout", "DecisionCase",
    "Batch", "StepStats", "TargetSlot", "SampleTargets",
    "BenchmarkReport", "CategoryHits",
    # Transforms & Paths
    "format_state",
    "list_weights",
    "DATA_DIR", "CORPUS_DIR", "RAW_DIR",
    "resolve_device",
]
