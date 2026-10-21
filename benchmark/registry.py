"""Benchmark registry for available decision engines and published suites (PFD)."""

from typing import Type
from benchmark.competitors.d1_omni import D1OmniEngine
from benchmark.competitors.gliner_decide import GlinerDecideEngine
from benchmark.competitors.jevk5_lite import JevK5LiteEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine
from benchmark.suites.fast_decisions import FastDecisionsSuite
from benchmark.suites.jevbench import JevBenchSuite
from benchmark.suites.laya_typed import LayaTypedDecisionsSuite
from benchmark.suites.suite import IBenchmarkSuite
from inference.decision_engine import DecisionEngine, IDecisionEngine

AVAILABLE_ENGINES: dict[str, Type[IDecisionEngine]] = {
    "piqo": DecisionEngine,
    "gliner": GlinerDecideEngine,
    "laya": LayaEngine,
    "verdict": VerdictEngine,
    "d1_omni": D1OmniEngine,
    "jevk5_lite": JevK5LiteEngine,
}

BENCHMARK_SUITES: dict[str, Type[IBenchmarkSuite]] = {
    "laya_typed_decisions": LayaTypedDecisionsSuite,
    "fast_decisions": FastDecisionsSuite,
    "jevbench_231": JevBenchSuite,
}
