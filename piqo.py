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
from training.epoch_trainer import ITrainer, EpochTrainer
from training.policies.schedule import ISchedule, CosineSchedule, ConstantSchedule
from training.telemetry.checkpointing_layer import CheckpointingLayer
from training.telemetry.metrics_layer import MetricsLayer
from training.stats import StepStats
from data.dataset import IDataset, Dataset, default_corpus_source
from inference.options import (
    IAnswer, ChoiceAnswer, NoulAnswer, ScoreAnswer,
    IOption, ChoiceOption, NoulOption, ScoreOption,
    choice_keys, choice_target_index, shuffle_choice,
    score_keys, score_target_index, matches_answer,
)
from model.policies.layout import format_state
from data.case import DataSplits, DecisionCase
from data.storage import IStoragePolicy, DiskStoragePolicy
from data.split.split import ISplitPolicy, BudgetSplitPolicy
from data.split.policies.stratification import IStratificationPolicy, RoundRobinStratification
from data.split.policies.leakage import ILeakagePolicy, JaccardLeakagePolicy
from data.sources.source import IDataSource, MultiSource
from data.sources.hf_cache import IRawCachePolicy, HFCache, DEFAULT_RAW_DIR
from data.sources.glaive_source import GlaiveSource
from data.sources.typed_decisions_source import TypedDecisionsSource
from data.sources.classification_source import ClassificationSource, TextClassificationSpec
from data.sources.multichoice_source import MultiChoiceSource, MultiChoiceSpec
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
    "IDataset", "Dataset", "default_corpus_source",
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
    "ISplitPolicy", "BudgetSplitPolicy", "IStratificationPolicy", "RoundRobinStratification", "ILeakagePolicy", "JaccardLeakagePolicy",
    "ILedgerPolicy", "DiskLedgerPolicy",
    "IScorerPolicy", "ScorerPolicy",
    "IReporterPolicy", "ConsoleReporterPolicy",
    "AVAILABLE_ENGINES", "BENCHMARK_SUITES",
    "IRawCachePolicy", "HFCache",
    "IDataSource", "MultiSource", "GlaiveSource", "TypedDecisionsSource", "ClassificationSource", "TextClassificationSpec", "MultiChoiceSource", "MultiChoiceSpec",
    # Layers (Lambda: P -> P)
    "ProfilingLayer", "CheckpointingLayer", "MetricsLayer",
    # Domain DTOs & Models
    "Weights",
    "IOption", "ChoiceOption", "NoulOption", "ScoreOption",
    "IAnswer", "ChoiceAnswer", "NoulAnswer", "ScoreAnswer",
    "TokenLayout", "PredictionLayout", "DecisionCase", "DataSplits",
    "Batch", "StepStats", "TokenSlot", "TargetSlot", "SampleTargets",
    "BenchmarkReport",
    # Transforms & Paths
    "format_state",
    "choice_keys",
    "choice_target_index",
    "shuffle_choice",
    "score_keys",
    "score_target_index",
    "matches_answer",
    "list_weights",
    "has_weights",
    "resolve_device",
    "resolve_vram_gb",
    "DEFAULT_RAW_DIR",
]
