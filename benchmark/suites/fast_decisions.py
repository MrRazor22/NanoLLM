from typing import Optional, Sequence
from benchmark.suites.case import BenchmarkCase
from benchmark.suites.manifest import IManifestPolicy, VerifiedManifestPolicy
from benchmark.suites.suite import IBenchmarkSuite
from inference.options import ChoiceAnswer, ChoiceOption

EXPECTED_SHA = "af9543ad6fb1607b010296151b32454154623ca9d1a92a7240fd02924c9e9931"

class FastDecisionsSuite(IBenchmarkSuite):
    def __init__(self, manifest: Optional[IManifestPolicy] = None):
        self._manifest = manifest or VerifiedManifestPolicy()
        self._cases: Optional[Sequence[BenchmarkCase]] = None

    def test_cases(self) -> Sequence[BenchmarkCase]:
        if self._cases is not None:
            return self._cases

        cases: list[BenchmarkCase] = []
        for record in self._manifest.load("fast_decisions.jsonl", EXPECTED_SHA):
            spec = record["questions"]["decision"]
            raw_crit = spec["criteria"]
            criteria = {str(k): str(v) for k, v in raw_crit.items()} if isinstance(raw_crit, dict) else {str(c): "" for c in raw_crit}
            target = str(record["gold"]["decision"]["label"])
            option = ChoiceOption(instruction=str(spec.get("instructions", "")), criteria=criteria)
            expected = ChoiceAnswer(choice=target, confidence=1.0, probabilities={c: (1.0 if c == target else 0.0) for c in criteria})
            cases.append(BenchmarkCase(state=str(record["state"]), option=option, expected=expected, category=str(record.get("category") or record["workflow"])))

        self._cases = cases
        return self._cases
