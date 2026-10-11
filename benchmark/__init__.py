"""Benchmark Domain: Sovereign Micro-Decision Evaluation & Benchmarking (PFD)."""

from benchmark.scorecard import BenchmarkItem, BenchmarkReport, CategoryHits, IScorecard, SCORES_FILE, Scorecard
from benchmark.competitors import AVAILABLE_COMPETITORS, LayaEngine, VerdictEngine

__all__ = [
    # Core Primitive
    "IScorecard",
    "Scorecard",
    # Competitor Adapters & Registry
    "AVAILABLE_COMPETITORS",
    "LayaEngine",
    "VerdictEngine",
    # Ledger & DTOs
    "SCORES_FILE",
    "CategoryHits",
    "BenchmarkItem",
    "BenchmarkReport",
]
