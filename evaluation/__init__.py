"""Evaluation Domain: Sovereign Model Quality & Benchmark Primitive (PFD)."""

from evaluation.policies.tracks import (
    TRACKS,
    AbstentionSource,
    AgenticSource,
    IBenchmarkSource,
    LayaSource,
    TypedDecisionsSource,
)
from evaluation.scorecard import (
    BenchmarkItem,
    BenchmarkReport,
    IEvaluator,
    Scorecard,
)

__all__ = [
    # Core Primitive
    "IEvaluator",
    "Scorecard",
    # DTOs
    "BenchmarkItem",
    "BenchmarkReport",
    # Policies & Tracks
    "IBenchmarkSource",
    "LayaSource",
    "AgenticSource",
    "AbstentionSource",
    "TypedDecisionsSource",
    "TRACKS",
]
