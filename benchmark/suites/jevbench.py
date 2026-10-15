from typing import Mapping, Sequence
from data.case import DecisionCase
from data.storage_layer import DiskStorageLayer
from benchmark.suites.suite import IBenchmarkSuite

class JevBenchSuite(IBenchmarkSuite):
    """TypeSafe AI / JevBench official benchmark tasks across hard reasoning and policy decisions."""

    def __init__(self, track: str = "hard"):
        self.track = track

    @property
    def name(self) -> str:
        return f"jevbench_{self.track}"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Jev-1.13": 0.845,
            "SemIf (Qwen3.5)": 0.732,
            "Laya (Zero-Shot)": 0.584,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        cases = DiskStorageLayer().load("test")
        reasoning_cats = {"causal_reasoning", "adversarial_grounding", "abductive_reasoning", "differential_elimination"}
        selected = [c for c in cases if c.category in reasoning_cats]
        return selected if selected else cases[:300]
