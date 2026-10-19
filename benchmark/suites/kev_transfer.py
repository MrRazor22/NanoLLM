from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase, case_from_dict
from benchmark.suites.suite import IBenchmarkSuite, load_verified_manifest

BENCHMARK_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"

class KevTransferSuite(IBenchmarkSuite):
    """Jared Palmer's official Kev out-of-domain transfer evaluation suite (SciQ, ANLI, Emotion, ARC)."""

    @property
    def name(self) -> str:
        return "kev_transfer"

    @property
    def manifest_hash(self) -> str:
        return "c30a91274f9b483aac9e4f02ada5dea953b3f1a2829456e0bc07806c4e73b517"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Kev-4B": 0.720,
            "Verdict-151M": 0.428,
            "GLiNER2.5-Decide": 0.404,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        import json
        url = "https://raw.githubusercontent.com/jaredpalmer/kev/main/evals/v4/transfer-v4/test.jsonl"
        content = load_verified_manifest(BENCHMARK_MANIFESTS_DIR / "kev_transfer_v4.jsonl", "c30a91274f9b483aac9e4f02ada5dea953b3f1a2829456e0bc07806c4e73b517", url)
        cases: list[DecisionCase] = []
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("category"):
                row["category"] = row.get("_meta", {}).get("source", "transfer")
            cases.append(case_from_dict(row, source=self.name))
        return cases
