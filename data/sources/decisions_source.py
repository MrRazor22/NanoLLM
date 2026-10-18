from dataclasses import replace
from typing import Sequence
from data.case import DecisionCase, case_from_dict
from data.sources.source import IDataSource
from data.sources.huggingface_source import load_raw_jsonl

class DecisionsSource(IDataSource):
    """Real Typed-Decisions dataset extractor: loads multi-question typed decision benchmarks."""
    name = "typed_decisions"

    def __init__(self, repeat: int = 5):
        self.repeat = repeat

    def read(self) -> Sequence[DecisionCase]:
        train_rows = load_raw_jsonl(f"{self.name}_train", "LocalLLaMA/typed-decisions", split="train", subset="all")
        test_rows = load_raw_jsonl(f"{self.name}_test", "LocalLLaMA/typed-decisions", split="test", subset="all")

        train_cases = [replace(case_from_dict(r), split="train", source=self.name) for r in train_rows] * self.repeat
        test_cases = [replace(case_from_dict(r), split="test", source=self.name) for r in test_rows]
        return train_cases + test_cases
