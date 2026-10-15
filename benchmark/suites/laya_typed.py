import json
from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase
from benchmark.suites.suite import IBenchmarkSuite

BENCHMARK_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class LayaTypedDecisionsSuite(IBenchmarkSuite):
    """Official LocalLLaMA/typed-decisions 400-case test benchmark (Laya, Jev, Verdict)."""

    @property
    def name(self) -> str:
        return "laya_typed_decisions"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Laya (Fine-Tuned)": 0.767,
            "Jev 1.13.0": 0.727,
            "Verdict-151M": 0.261,
            "Laya (Zero-Shot)": 0.362,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        path = BENCHMARK_DATA_DIR / "typed_decisions_test.jsonl"
        if not path.exists():
            from datasets import load_dataset
            rows = list(load_dataset("LocalLLaMA/typed-decisions", split="test"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")

        with open(path, "r", encoding="utf-8") as f:
            return [DecisionCase.from_dict(json.loads(line)) for line in f if line.strip()]
