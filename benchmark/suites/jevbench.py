from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase
from benchmark.suites.suite import IBenchmarkSuite, load_verified_manifest, adapt_manifest_case

BENCHMARK_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"

class JevBenchSuite(IBenchmarkSuite):
    """TypeSafe AI / JevBench official 231 public evaluation tasks (Easy 48, Standard 72, Hard 111)."""

    @property
    def name(self) -> str:
        return "jevbench_231"

    @property
    def manifest_hash(self) -> str:
        return "d2677f4d6f1e2063657183a36f4dbd92c0383136c050084701e2ee3d6ccc49a6"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Jev-1.13": 0.845,
            "SemIf (Qwen3.5)": 0.732,
            "Verdict v1.4": 0.646,
            "JevK5-Lite": 0.628,
            "Laya (Zero-Shot)": 0.584,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        import json
        content = load_verified_manifest(BENCHMARK_MANIFESTS_DIR / "jevbench_231.jsonl", "d2677f4d6f1e2063657183a36f4dbd92c0383136c050084701e2ee3d6ccc49a6")
        return [adapt_manifest_case(json.loads(line), source=self.name) for line in content.decode("utf-8").splitlines() if line.strip()]
