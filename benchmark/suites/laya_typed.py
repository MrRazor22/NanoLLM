from typing import Mapping, Sequence
from data.case import DecisionCase
from benchmark.suites.suite import IBenchmarkSuite

class LayaTypedDecisionsSuite(IBenchmarkSuite):
    """Laya & Jev flagship published benchmark: LocalLLaMA/typed-decisions (400 cases)."""

    @property
    def name(self) -> str:
        return "laya_typed_decisions"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Laya (Fine-Tuned)": 0.767,
            "Jev 1.13.0": 0.727,
            "Laya (Zero-Shot)": 0.362,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        from data.sources.decisions_source import DecisionsSource
        src = DecisionsSource(repeat=1)
        records = src.read()
        return [r.sample for r in records if r.tag == "test"]
