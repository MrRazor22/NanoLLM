"""Benchmark registry for available decision engines and published suites (PFD)."""

from typing import Type
from benchmark.competitors.gliner_decide import GlinerDecideEngine
from benchmark.competitors.jev import JevEngine
from benchmark.competitors.kev import KevEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine
from benchmark.suites.fast_decisions import FastDecisionsSuite
from benchmark.suites.jevbench import JevBenchSuite
from benchmark.suites.kev_transfer import KevTransferSuite
from benchmark.suites.laya_typed import LayaTypedDecisionsSuite
from benchmark.suites.verdict_nli import VerdictNliSuite
from benchmark.suites.suite import IBenchmarkSuite
from inference.decision_engine import DecisionEngine, IDecisionEngine

AVAILABLE_ENGINES: dict[str, Type[IDecisionEngine]] = {
    "piqo": DecisionEngine,
    "gliner": GlinerDecideEngine,
    "jev": JevEngine,
    "kev": KevEngine,
    "laya": LayaEngine,
    "verdict": VerdictEngine,
}

OFFICIAL_SUITES: dict[str, Type[IBenchmarkSuite]] = {
    "laya": LayaTypedDecisionsSuite,
    "laya_typed": LayaTypedDecisionsSuite,
    "laya_typed_decisions": LayaTypedDecisionsSuite,
    "typed_decisions": LayaTypedDecisionsSuite,
}

PUBLISHED_SUITES: dict[str, Type[IBenchmarkSuite]] = {
    **OFFICIAL_SUITES,
    "gliner": FastDecisionsSuite,
    "fastino": FastDecisionsSuite,
    "fast_decisions": FastDecisionsSuite,
    "jev": JevBenchSuite,
    "jevbench": JevBenchSuite,
    "kev": KevTransferSuite,
    "kev_transfer": KevTransferSuite,
    "verdict": VerdictNliSuite,
    "verdict_nli": VerdictNliSuite,
}
