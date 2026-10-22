from typing import Optional, Sequence
from benchmark.suites.case import BenchmarkCase
from benchmark.suites.manifest import IManifestPolicy, VerifiedManifestPolicy
from benchmark.suites.suite import IBenchmarkSuite
from inference.options import ChoiceAnswer, ChoiceOption

EXPECTED_SHA = "d2677f4d6f1e2063657183a36f4dbd92c0383136c050084701e2ee3d6ccc49a6"

class JevBenchSuite(IBenchmarkSuite):
    def __init__(self, manifest: Optional[IManifestPolicy] = None):
        self._manifest = manifest or VerifiedManifestPolicy()
        self._cases: Optional[Sequence[BenchmarkCase]] = None

    def test_cases(self) -> Sequence[BenchmarkCase]:
        if self._cases is not None:
            return self._cases

        cases: list[BenchmarkCase] = []
        for record in self._manifest.load("jevbench_231.jsonl", EXPECTED_SHA):
            spec = record["questions"]["decision"]
            criteria = {str(k): str(v) for k, v in spec["criteria"].items()}
            target = str(record["gold"]["decision"])
            option = ChoiceOption(instruction=str(spec.get("instructions", "")), criteria=criteria)
            expected = ChoiceAnswer(choice=target, confidence=1.0, probabilities={c: (1.0 if c == target else 0.0) for c in criteria})
            cases.append(BenchmarkCase(state=str(record["state"]), option=option, expected=expected, category=str(record.get("category") or record["workflow"])))

        self._cases = cases
        return self._cases
