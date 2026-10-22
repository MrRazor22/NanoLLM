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
            category = str(record.get("category") or record.get("workflow") or "benchmark")
            split = record.get("split")

            questions = json.loads(record["questions"]) if isinstance(record.get("questions"), str) else record["questions"]
            gold = json.loads(record["gold"]) if isinstance(record.get("gold"), str) else (record.get("gold") or {})

            _, spec = next(iter(questions.items()))
            criteria = spec.get("options") or spec.get("criteria") or ("false", "true")
            candidates = tuple(str(key) for key in (criteria.keys() if isinstance(criteria, dict) else criteria))
            instruction = str(spec.get("instructions") or spec.get("instruction") or "")

            gold_item = next(iter(gold.values())) if isinstance(gold, dict) and gold else gold
            raw_probs = gold_item.get("probabilities") if isinstance(gold_item, dict) else None

            probabilities: dict[str, float] = {}
            if raw_probs and isinstance(raw_probs, dict):
                probabilities = {
                    candidate: float(raw_probs.get(candidate, raw_probs.get(str(i), 0.0)))
                    for i, candidate in enumerate(candidates)
                }

            if sum(probabilities.values()) == 0.0:
                raw_label = (gold_item.get("label") if isinstance(gold_item, dict) else gold_item) if gold_item else spec.get("label")
                label = str(raw_label or "").strip()
                if label.isdigit() and 0 <= int(label) < len(candidates) and label not in candidates:
                    label = candidates[int(label)]
                probabilities = {candidate: (1.0 if candidate == label or candidate.lower() == label.lower() else 0.0) for candidate in candidates}

            if sum(probabilities.values()) == 0.0:
                raise ValueError(f"Target label not found in candidates {candidates}")

            cases.append(
                DecisionCase(
                    state=state,
                    candidates=candidates,
                    instruction=instruction,
                    probabilities=probabilities,
                    meta=CaseMeta(category=category, source="laya_typed_decisions", split=split),
                )
            )

        self._cases = cases
        return self._cases
