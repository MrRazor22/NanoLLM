from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase
from benchmark.suites.suite import IBenchmarkSuite

BENCHMARK_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class FastDecisionsSuite(IBenchmarkSuite):
    """Fastino Labs GLiNER2.5-Decide published 17-task operational benchmark."""

    @property
    def name(self) -> str:
        return "fast_decisions"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "GLiNER2.5-Decide": 0.601,
            "JevK5": 0.575,
            "Laya": 0.570,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        path = BENCHMARK_DATA_DIR / "fast_decisions_test.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Official Fastino 17-task test set not found at '{path}'. "
                "Slicing local training data is strictly prohibited under PFD. "
                "Provide the official frozen dataset to run this benchmark."
            )
        import json
        with open(path, "r", encoding="utf-8") as f:
            return [DecisionCase(**json.loads(line)) for line in f if line.strip()]
