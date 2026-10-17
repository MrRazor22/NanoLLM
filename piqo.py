"""Piqo: Sovereign Composition Root (PFD)."""

from model.torch_model import IModel, TorchModel
from model.policies.calibrated_loss import ILoss, CalibratedLoss
from model.policies.head import IHead, MLPHead
from model.policies.optimizer import IOptimizerPolicy, AdamWOptimizer
from model.policies.layout import ILayout, MaskedLayout
from model.policies.weight_cache import (
    DEFAULT_WEIGHTS_DIR, DiskWeightCache, IWeightCache,
)
from model.device import resolve_device, resolve_vram_gb
from model.weights import Weights, WeightMeta, list_weights, has_weights
from model.token_layout import TokenLayout, PredictionLayout, TokenSlot, Batch, SampleTargets, TargetSlot
from inference.decision_engine import IDecisionEngine, DecisionEngine
from inference.decision_resolver import IResolver, DecisionResolver
from inference.profiling_layer import ProfilingLayer
from inference.questions import (
    DecisionState, Choice, Noul, Score, Question,
    IAnswer, ChoiceResult, NoulResult, ScoreResult, Answer, DecisionResult,
    format_state, choice_keys, choice_target_index, shuffle_choice,
    score_to_norm, score_from_norm, matches_answer,
)
from training.epoch_trainer import ITrainer, EpochTrainer
from training.policies.schedule import ISchedule, CosineSchedule, ConstantSchedule
from training.telemetry.checkpointing_layer import CheckpointingLayer
from training.telemetry.metrics_layer import MetricsLayer
from training.stats import StepStats
from data.dataset import IDataset, Dataset
from data.case import DataSplits, SourceRecord, DecisionCase, case_to_dict, case_from_dict, case_gold
from data.storage_policy import IStoragePolicy, DiskStoragePolicy
from data.curation.budget_split import ISplitPolicy, BudgetSplitPolicy
from data.curation.ngram_leakage import ILeakagePolicy, NGramLeakagePolicy
from data.sources.source import IDataSource
from data.registry import AVAILABLE_SOURCES
from data.sources.glaive_source import GlaiveSource
from data.sources.decisions_source import DecisionsSource
from data.sources.huggingface_source import HuggingFaceSource, BenchmarkSpec, RAW_DIR
from benchmark.policies.scorer import BenchmarkReport, IScorerPolicy, ScorerPolicy
from benchmark.policies.ledger import ILedgerPolicy, DiskLedgerPolicy
from benchmark.policies.reporter import IReporterPolicy, ConsoleReporterPolicy
from benchmark.benchmark import IBenchmark, Benchmark
from benchmark.registry import AVAILABLE_ENGINES, BENCHMARK_SUITES
from benchmark.competitors.gliner_decide import GlinerDecideEngine
from benchmark.competitors.jev import JevEngine
from benchmark.competitors.kev import KevEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine

__all__ = [
    # Core Primitives
    "IModel", "TorchModel",
    "IDecisionEngine", "DecisionEngine",
    "ITrainer", "EpochTrainer",
    "IDataset", "Dataset",
    "IBenchmark", "Benchmark", "GlinerDecideEngine", "JevEngine", "KevEngine", "LayaEngine", "VerdictEngine",
    # Policies & Injected Sub-Primitives
    "IHead", "MLPHead",
    "IOptimizerPolicy", "AdamWOptimizer",
    "ILoss", "CalibratedLoss",
    "ILayout", "MaskedLayout",
    "IWeightCache", "DiskWeightCache", "DEFAULT_WEIGHTS_DIR",
    "ISchedule", "CosineSchedule", "ConstantSchedule",
    "IResolver", "DecisionResolver",
    "IStoragePolicy", "DiskStoragePolicy",
    "ISplitPolicy", "BudgetSplitPolicy", "ILeakagePolicy", "NGramLeakagePolicy",
    "ILedgerPolicy", "DiskLedgerPolicy",
    "IScorerPolicy", "ScorerPolicy",
    "IReporterPolicy", "ConsoleReporterPolicy",
    "AVAILABLE_ENGINES", "BENCHMARK_SUITES", "AVAILABLE_SOURCES",
    "IDataSource", "GlaiveSource", "DecisionsSource", "HuggingFaceSource", "BenchmarkSpec",
    # Layers (Lambda: P -> P)
    "ProfilingLayer", "CheckpointingLayer", "MetricsLayer",
    # Domain DTOs & Models
    "Weights",
    "DecisionState", "Choice", "Noul", "Score", "Question",
    "IAnswer", "ChoiceResult", "NoulResult", "ScoreResult", "Answer", "DecisionResult",
    "TokenLayout", "PredictionLayout", "DecisionCase", "DataSplits", "SourceRecord",
    "Batch", "StepStats", "TokenSlot", "TargetSlot", "SampleTargets",
    "BenchmarkReport",
    # Transforms & Paths
    "format_state",
    "choice_keys",
    "choice_target_index",
    "shuffle_choice",
    "score_to_norm",
    "score_from_norm",
    "matches_answer",
    "case_to_dict",
    "case_from_dict",
    "case_gold",
    "list_weights",
    "DEFAULT_WEIGHTS_DIR", "RAW_DIR",
    "resolve_device",
]
