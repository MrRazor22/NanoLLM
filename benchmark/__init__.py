"""Benchmark Domain: Sovereign Micro-Decision Evaluation & Benchmarking (PFD)."""

from benchmark.scorecard import BenchmarkItem, BenchmarkReport, CategoryHits, IScorecard, SCORES_FILE, Scorecard
from benchmark.ledger import DiskLedgerPolicy, ILedgerPolicy
from benchmark.competitors import AVAILABLE_COMPETITORS, GlinerDecideEngine, KevEngine, LayaEngine, VerdictEngine

__all__ = [
    # Core Primitive
    "IScorecard",
    "Scorecard",
    # Injected Opinion Sub-Primitives
    "ILedgerPolicy",
    "DiskLedgerPolicy",
    # Competitor Adapters & Registry
    "AVAILABLE_COMPETITORS",
    "GlinerDecideEngine",
    "KevEngine",
    "LayaEngine",
    "VerdictEngine",
    # DTOs
    "SCORES_FILE",
    "CategoryHits",
    "BenchmarkItem",
    "BenchmarkReport",
]
