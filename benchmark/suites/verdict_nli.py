from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase
from benchmark.suites.suite import IBenchmarkSuite

BENCHMARK_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class VerdictNliSuite(IBenchmarkSuite):
    """Verdict multi-candidate decision benchmark.

    Note: Verdict's published baseline (0.261) was officially established on
    LocalLLaMA/typed-decisions (test split); see LayaTypedDecisionsSuite.
    """

    @property
    def name(self) -> str:
        return "verdict_nli"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Verdict-151M": 0.261,
            "GLiClass-ModernBERT": 0.240,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        path = BENCHMARK_DATA_DIR / "verdict_nli_test.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Official Verdict evaluation manifest not found at '{path}'. "
                "Note: Verdict's official published evaluation was established on "
                "LocalLLaMA/typed-decisions (run with '-s laya_typed' instead). "
                "Slicing local training data is strictly prohibited under PFD."
            )
        import json
        with open(path, "r", encoding="utf-8") as f:
            return [DecisionCase(**json.loads(line)) for line in f if line.strip()]
