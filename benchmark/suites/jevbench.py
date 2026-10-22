from typing import Optional, Sequence
from benchmark.suites.manifest import IManifestPolicy, VerifiedManifestPolicy
from benchmark.suites.suite import IBenchmarkSuite
from data.case import CaseMeta, DecisionCase

EXPECTED_SHA = "d2677f4d6f1e2063657183a36f4dbd92c0383136c050084701e2ee3d6ccc49a6"

class JevBenchSuite(IBenchmarkSuite):
    def __init__(self, manifest: Optional[IManifestPolicy] = None):
        self._manifest = manifest or VerifiedManifestPolicy()
        self._cases: Optional[Sequence[DecisionCase]] = None

    def test_cases(self) -> Sequence[DecisionCase]:
        if self._cases is not None:
            return self._cases

        cases: list[DecisionCase] = []
        for record in self._manifest.load("jevbench_231.jsonl", EXPECTED_SHA):
            state = str(record.get("state", ""))

            spec = record["questions"]["decision"]
            candidates = tuple(str(option) for option in spec["criteria"].keys())

            instruction = str(spec.get("instructions") or spec.get("instruction") or "")

            target = str(record["gold"]["decision"])
            probabilities = {candidate: (1.0 if candidate == target else 0.0) for candidate in candidates}

            category = str(record.get("category") or record.get("workflow") or "benchmark")
            split = record.get("split")
            meta = CaseMeta(category=category, source="jevbench_231", split=split)

            cases.append(
                DecisionCase(
                    state=state,
                    candidates=candidates,
                    instruction=instruction,
                    probabilities=probabilities,
                    meta=meta,
                )
            )

        self._cases = cases
        return self._cases
