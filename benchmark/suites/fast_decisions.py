from typing import Optional, Sequence
from benchmark.suites.manifest import IManifestPolicy, VerifiedManifestPolicy
from benchmark.suites.suite import IBenchmarkSuite
from data.case import CaseMeta, DecisionCase

EXPECTED_SHA = "af9543ad6fb1607b010296151b32454154623ca9d1a92a7240fd02924c9e9931"

class FastDecisionsSuite(IBenchmarkSuite):
    def __init__(self, manifest: Optional[IManifestPolicy] = None):
        self._manifest = manifest or VerifiedManifestPolicy()
        self._cases: Optional[Sequence[DecisionCase]] = None

    def test_cases(self) -> Sequence[DecisionCase]:
        if self._cases is not None:
            return self._cases

        cases: list[DecisionCase] = []
        for record in self._manifest.load("fast_decisions.jsonl", EXPECTED_SHA):
            state = str(record.get("state", ""))

            spec = record["questions"]["decision"]
            candidates = tuple(str(option) for option in spec["criteria"])

            instruction = str(spec.get("instructions") or spec.get("instruction") or "")

            target = str(record["gold"]["decision"]["label"])
            probabilities = {candidate: (1.0 if candidate == target else 0.0) for candidate in candidates}

            category = str(record.get("category") or record.get("workflow") or "benchmark")
            split = record.get("split")
            meta = CaseMeta(category=category, source="fast_decisions", split=split)

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
