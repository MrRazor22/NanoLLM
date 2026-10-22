from typing import Mapping, Sequence, Type
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
from inference.profiling_layer import ProfilingLayer

SUITE_ALIASES = {"jev": "jevbench_231", "fast": "fast_decisions", "laya": "laya_typed_decisions"}
ENGINE_ALIASES = {"d1": "d1_omni", "jevk5": "jevk5_lite"}

SUITES: dict[str, Type[IBenchmarkSuite]] = {
    "jevbench_231": JevBenchSuite,
    "fast_decisions": FastDecisionsSuite,
    "laya_typed_decisions": LayaTypedDecisionsSuite,
}

ENGINES: dict[str, Type[IDecisionEngine]] = {
    "piqo": DecisionEngine,
    "gliner": GlinerDecideEngine,
    "laya": LayaEngine,
    "verdict": VerdictEngine,
    "d1_omni": D1OmniEngine,
    "jevk5_lite": JevK5LiteEngine,
}

DEFAULT_ENGINES = ["piqo", "gliner", "laya", "verdict"]

def load_suites(keys: Sequence[str] | str = "all") -> Mapping[str, IBenchmarkSuite]:
    raw = list(SUITES.keys()) if keys == "all" else [k.strip().lower() for k in (keys.split(",") if isinstance(keys, str) else keys)]
    names = [SUITE_ALIASES.get(r, r) for r in raw]
    for n in names:
        if n not in SUITES:
            raise KeyError(f"Unknown suite '{n}'. Available: {list(SUITES.keys())}")
    return {n: SUITES[n]() for n in names}

def load_engines(keys: Sequence[str] | str = "all") -> Mapping[str, IDecisionEngine]:
    raw = DEFAULT_ENGINES if keys in ("all", "default") else [k.strip().lower() for k in (keys.split(",") if isinstance(keys, str) else keys)]
    names = [ENGINE_ALIASES.get(r, r) for r in raw]
    for n in names:
        if n not in ENGINES:
            raise KeyError(f"Unknown engine '{n}'. Available: {list(ENGINES.keys())}")
    return {n.title(): ENGINES[n]() | ProfilingLayer() for n in names}
