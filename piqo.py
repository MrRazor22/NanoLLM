"""Piqo: Sovereign Composition Root (PFD)."""

from model.torch_model import IModel, TorchModel
from model.layout import ILayout, MaskedLayout
from model.weights import WeightInfo, list_weights
from model.device import resolve_device

from inference.decision_engine import IDecisionEngine, DecisionEngine
from inference.profiling_layer import ProfilingLayer
from inference.options import (
    IAnswer, ChoiceAnswer, NoulAnswer, ScoreAnswer,
    IOption, ChoiceOption, NoulOption, ScoreOption,
)

from training.epoch_trainer import ITrainer, EpochTrainer
from training.checkpointing_layer import CheckpointingLayer, TrainingStats
from training.policies.loss import ILoss, CalibratedLoss
from training.policies.learner import ILearner, AdamWLearner
from training.policies.schedule import ISchedule, CosineSchedule

from data.dataset import IDataset, Dataset
from data.case import DecisionCase
from data.storage import IStoragePolicy, DiskStoragePolicy
from data.split.split import ISplitPolicy, BudgetSplitPolicy
from data.split.policies.stratification import IStratificationPolicy, RoundRobinStratification
from data.split.policies.leakage import ILeakagePolicy, JaccardLeakagePolicy
from data.sources.source import IDataSource, MultiSource
from data.sources.hf_cache import IRawCachePolicy, HFCache
from data.sources.glaive_source import GlaiveSource
from data.sources.typed_decisions_source import TypedDecisionsSource
from data.sources.classification_source import ClassificationSource
from data.sources.multichoice_source import MultiChoiceSource

from benchmark.benchmark import IBenchmark, Benchmark
from benchmark.policies.scorer import BenchmarkReport, IScorerPolicy, ScorerPolicy
from benchmark.policies.scorecard import IScorecardPolicy, DiskScorecardPolicy
from benchmark.policies.reporter import IReporterPolicy, ConsoleReporterPolicy

__all__ = [
    # Core Primitives
    "IModel", "TorchModel",
    "IDecisionEngine", "DecisionEngine",
    "ITrainer", "EpochTrainer",
    "IDataset", "Dataset",
    "IBenchmark", "Benchmark",
    # Inference Options & Answers
    "IOption", "ChoiceOption", "NoulOption", "ScoreOption",
    "IAnswer", "ChoiceAnswer", "NoulAnswer", "ScoreAnswer",
    # Data Models & Sources
    "DecisionCase",
    "IDataSource", "MultiSource", "GlaiveSource", "TypedDecisionsSource", "ClassificationSource", "MultiChoiceSource",
    # Policies & Composition
    "ILayout", "MaskedLayout",
    "ILearner", "AdamWLearner",
    "ILoss", "CalibratedLoss",
    "ISchedule", "CosineSchedule",
    "IStoragePolicy", "DiskStoragePolicy",
    "ISplitPolicy", "BudgetSplitPolicy",
    "IStratificationPolicy", "RoundRobinStratification",
    "ILeakagePolicy", "JaccardLeakagePolicy",
    "IRawCachePolicy", "HFCache",
    "IScorecardPolicy", "DiskScorecardPolicy",
    "IScorerPolicy", "ScorerPolicy",
    "IReporterPolicy", "ConsoleReporterPolicy",
    # Layers (Lambda: P -> P)
    "ProfilingLayer", "CheckpointingLayer",
    # Inspection & DTOs
    "WeightInfo", "TrainingStats", "BenchmarkReport",
    "list_weights", "resolve_device",
]
