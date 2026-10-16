from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase
from benchmark.suites.suite import IBenchmarkSuite, load_verified_manifest

BENCHMARK_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"

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
        import pyarrow.parquet as pq
        path = BENCHMARK_MANIFESTS_DIR / "typed_decisions_test.parquet"
        url = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all/test-00000-of-00001.parquet"
        load_verified_manifest(path, "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c", url)
        return [DecisionCase.from_dict(r) for r in pq.read_table(path).to_pylist()]
