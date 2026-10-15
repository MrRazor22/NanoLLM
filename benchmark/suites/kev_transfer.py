from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase
from benchmark.suites.suite import IBenchmarkSuite

BENCHMARK_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class KevTransferSuite(IBenchmarkSuite):
    """Jared Palmer's official Kev out-of-domain transfer evaluation suite (SciQ, ANLI, Emotion, ARC)."""

    @property
    def name(self) -> str:
        return "kev_transfer"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Kev-4B": 0.720,
            "Verdict-151M": 0.428,
            "GLiNER2.5-Decide": 0.404,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        path = BENCHMARK_DATA_DIR / "kev_transfer_v4.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Official Kev transfer-v4 manifest not found at '{path}'. "
                "Slicing local training data is strictly prohibited under PFD. "
                "Reproduce Kev's frozen transfer-v4 suite and manifest to run."
            )
        import json
        with open(path, "r", encoding="utf-8") as f:
            return [DecisionCase(**json.loads(line)) for line in f if line.strip()]
