from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase
from benchmark.suites.suite import IBenchmarkSuite, load_verified_manifest, adapt_manifest_case

BENCHMARK_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"

class FastDecisionsSuite(IBenchmarkSuite):
    """Fastino Labs / Liquid AI Fast Decisions 17-task operational benchmark (1,700 cases)."""

    @property
    def name(self) -> str:
        return "fast_decisions"

    @property
    def manifest_hash(self) -> str:
        return "af9543ad6fb1607b010296151b32454154623ca9d1a92a7240fd02924c9e9931"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "d1-omni-600M": 0.769,
            "GLiNER2.5-Decide": 0.637,
            "JevK5-Lite": 0.587,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        import json
        content = load_verified_manifest(BENCHMARK_MANIFESTS_DIR / "fast_decisions.jsonl", self.manifest_hash)
        return [adapt_manifest_case(json.loads(line), source=self.name) for line in content.decode("utf-8").splitlines() if line.strip()]
