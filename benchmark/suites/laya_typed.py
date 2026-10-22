import json
from typing import Optional, Sequence
from benchmark.suites.case import BenchmarkCase
from benchmark.suites.manifest import IManifestPolicy, VerifiedManifestPolicy
from benchmark.suites.suite import IBenchmarkSuite
from inference.options import ChoiceAnswer, ChoiceOption

EXPECTED_SHA = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
SOURCE_URL = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all/test-00000-of-00001.parquet"

class LayaTypedDecisionsSuite(IBenchmarkSuite):
    def __init__(self, manifest: Optional[IManifestPolicy] = None):
        self._manifest = manifest or VerifiedManifestPolicy()
        self._cases: Optional[Sequence[BenchmarkCase]] = None

    def test_cases(self) -> Sequence[BenchmarkCase]:
        if self._cases is not None:
            return self._cases

        cases: list[BenchmarkCase] = []
        for record in self._manifest.load("typed_decisions_test.parquet", EXPECTED_SHA, url=SOURCE_URL):
            questions = json.loads(record["questions"])
            gold = json.loads(record["gold"])
            q_name, spec = next(iter(questions.items()))
            gold_spec = gold[q_name]

            raw_crit = spec.get("criteria") or gold_spec["probabilities"]
            criteria = {str(k): str(v) for k, v in raw_crit.items()} if isinstance(raw_crit, dict) else {str(c): "" for c in raw_crit}
            probabilities = {str(k): float(v) for k, v in gold_spec["probabilities"].items()}
            target = max(probabilities, key=probabilities.__getitem__)

            option = ChoiceOption(instruction=str(spec.get("instructions", "")), criteria=criteria)
            expected = ChoiceAnswer(choice=target, confidence=float(probabilities[target]), probabilities=probabilities)
            cases.append(BenchmarkCase(state=str(record["state"]), option=option, expected=expected, category=str(record.get("category") or record["workflow"])))

        self._cases = cases
        return self._cases
