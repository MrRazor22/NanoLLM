from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase
from benchmark.suites.suite import IBenchmarkSuite

BENCHMARK_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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
        path = BENCHMARK_DATA_DIR / f"jevbench_{self.track}_test.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Official JevBench {self.track} benchmark records not found at '{path}'. "
                "Slicing local training data is strictly prohibited under PFD. "
                "Obtain official benchmark records to claim official reproduction."
            )
        import json
        with open(path, "r", encoding="utf-8") as f:
            return [DecisionCase(**json.loads(line)) for line in f if line.strip()]
