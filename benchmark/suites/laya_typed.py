import json
from typing import Optional, Sequence
from benchmark.suites.manifest import IManifestPolicy, VerifiedManifestPolicy
from benchmark.suites.suite import IBenchmarkSuite
from data.case import CaseMeta, DecisionCase

EXPECTED_SHA = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
SOURCE_URL = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all/test-00000-of-00001.parquet"

class LayaTypedDecisionsSuite(IBenchmarkSuite):
    def __init__(self, manifest: Optional[IManifestPolicy] = None):
        self._manifest = manifest or VerifiedManifestPolicy()
        self._cases: Optional[Sequence[DecisionCase]] = None

    def test_cases(self) -> Sequence[DecisionCase]:
        if self._cases is not None:
            return self._cases

        cases: list[DecisionCase] = []
        for record in self._manifest.load("typed_decisions_test.parquet", EXPECTED_SHA, url=SOURCE_URL):
            state = str(record.get("state", ""))

            questions = json.loads(record["questions"])
            gold = json.loads(record["gold"])

            q_name, spec = next(iter(questions.items()))
            gold_spec = gold[q_name]

            candidates = tuple(str(key) for key in gold_spec["probabilities"].keys())

            instruction = str(spec.get("instructions") or spec.get("instruction") or "")

            probabilities = {candidate: float(gold_spec["probabilities"][candidate]) for candidate in candidates}

            category = str(record.get("category") or record.get("workflow") or "benchmark")
            split = record.get("split")
            meta = CaseMeta(category=category, source="laya_typed_decisions", split=split)

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
