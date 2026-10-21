"""Piqo: Sovereign Composition Root (PFD)."""

from model.torch_model import IModel, TorchModel
from model.weights import list_weights, WeightInfo, DEFAULT_WEIGHTS_DIR, DEFAULT_WEIGHTS_PATH
from training.policies.loss import ILoss, CalibratedLoss
from training.policies.learner import ILearner, AdamWLearner
from model.layout import ILayout, MaskedLayout, Layout
from model.device import resolve_device, resolve_vram_gb, synchronize_device
from inference.decision_engine import IDecisionEngine, DecisionEngine
from inference.profiling_layer import ProfilingLayer
from training.epoch_trainer import ITrainer, EpochTrainer
from training.policies.schedule import ISchedule, CosineSchedule
from training.checkpointing_layer import CheckpointingLayer, TrainingStats
from data.dataset import IDataset, Dataset, default_corpus_source
from inference.options import (
    IAnswer, ChoiceAnswer, NoulAnswer, ScoreAnswer,
    IOption, ChoiceOption, NoulOption, ScoreOption,
    option_to_candidates, scores_to_answer,
)
from data.case import CaseMeta, DataSplits, DecisionCase
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
    "ILearner", "AdamWLearner",
    "ILoss", "CalibratedLoss",
    "ILayout", "MaskedLayout",
    "ISchedule", "CosineSchedule",
    "IStoragePolicy", "DiskStoragePolicy",
    "ISplitPolicy", "BudgetSplitPolicy", "IStratificationPolicy", "RoundRobinStratification", "ILeakagePolicy", "JaccardLeakagePolicy",
    "ILedgerPolicy", "DiskLedgerPolicy",
    "IScorerPolicy", "ScorerPolicy",
    "IReporterPolicy", "ConsoleReporterPolicy",
    "AVAILABLE_ENGINES", "BENCHMARK_SUITES",
    "IRawCachePolicy", "HFCache",
    "IDataSource", "MultiSource", "GlaiveSource", "TypedDecisionsSource", "ClassificationSource", "TextClassificationSpec", "MultiChoiceSource", "MultiChoiceSpec",
    # Layers (Lambda: P -> P)
    "ProfilingLayer", "CheckpointingLayer",
    # Domain DTOs & Models
    "WeightInfo", "TrainingStats",
    "IOption", "ChoiceOption", "NoulOption", "ScoreOption",
    "IAnswer", "ChoiceAnswer", "NoulAnswer", "ScoreAnswer",
    "DecisionCase", "CaseMeta", "DataSplits",
    "Layout",
    "BenchmarkReport",
    # Transforms & Paths
    "option_to_candidates",
    "scores_to_answer",
    "list_weights",
    "resolve_device",
    "resolve_vram_gb",
    "synchronize_device",
    "DEFAULT_WEIGHTS_DIR",
    "DEFAULT_WEIGHTS_PATH",
    "DEFAULT_RAW_DIR",
]
