from typing import Mapping, Sequence
from data.case import DecisionCase
from data.dataset import Dataset
from benchmark.suites.suite import IBenchmarkSuite

class CorpusHoldoutSuite(IBenchmarkSuite):
    """In-distribution evaluation holdout from dataset preparation."""

    def __init__(self, split: str = "test"):
        self.split = split

    @property
    def name(self) -> str:
        return f"corpus_{self.split}"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {}

    def load_cases(self) -> Sequence[DecisionCase]:
        return Dataset().load(self.split)
