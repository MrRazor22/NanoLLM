"""Benchmark registry for available decision engines and published suites (PFD)."""

from typing import Type
from benchmark.competitors.gliner_decide import GlinerDecideEngine
from benchmark.competitors.jev import JevEngine
from benchmark.competitors.kev import KevEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine
from benchmark.suites.gliner_fast_decisions import GlinerFastDecisionsSuite
from benchmark.suites.jev_bench import JevBenchSuite
from benchmark.suites.kev_transfer import KevTransferSuite
from benchmark.suites.laya_typed import LayaTypedDecisionsSuite
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

PUBLISHED_SUITES: dict[str, Type[IBenchmarkSuite]] = {
    "gliner": GlinerFastDecisionsSuite,
    "fastino": GlinerFastDecisionsSuite,
    "gliner_fast_decisions": GlinerFastDecisionsSuite,
    "jev": JevBenchSuite,
    "jevbench": JevBenchSuite,
    "jev_bench": JevBenchSuite,
    "kev": KevTransferSuite,
    "kev_transfer": KevTransferSuite,
    "laya": LayaTypedDecisionsSuite,
    "laya_typed": LayaTypedDecisionsSuite,
    "verdict": LayaTypedDecisionsSuite,
}
