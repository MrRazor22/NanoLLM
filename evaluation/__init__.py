"""Evaluation Domain: Sovereign Model Quality & Benchmark Primitive (PFD)."""

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
]
