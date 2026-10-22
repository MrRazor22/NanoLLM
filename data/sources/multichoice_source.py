from dataclasses import dataclass
from typing import List, Optional, Sequence
from data.case import CaseMeta, DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

@dataclass(frozen=True, slots=True)
class MultiChoiceSpec:
    dataset: str
    category: str
    subset: Optional[str] = None
    question_field: str = "question"
    choices_field: str = "choices"
    answer_field: str = "answerKey"
    splits: Sequence[str] = ("train", "test")

STANDARD_MULTICHOICE: Sequence[MultiChoiceSpec] = (
    MultiChoiceSpec("allenai/ai2_arc", "deductive_reasoning", subset="ARC-Challenge"),
    MultiChoiceSpec("allenai/openbookqa", "causal_reasoning", subset="main", question_field="question_stem"),
    MultiChoiceSpec("tau/commonsense_qa", "commonsense_reasoning"),
    MultiChoiceSpec("allenai/qasc", "multihop_reasoning"),
)

class MultiChoiceSource(IDataSource):
    def __init__(
        self,
        spec: Optional[MultiChoiceSpec] = None,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
        cache: Optional[IRawCachePolicy] = None,
    ):
        self.spec = spec or STANDARD_MULTICHOICE[0]
        self.train_limit = train_limit
        self.test_limit = test_limit
        self.cache = cache or HFCache()

    def cases(self) -> Sequence[DecisionCase]:
        source_tag = f"{self.spec.dataset.replace('/', '_')}_{self.spec.subset}" if self.spec.subset else self.spec.dataset.replace("/", "_")
        cases: List[DecisionCase] = []

        for split_name in self.spec.splits:
            limit = self.test_limit if split_name == "test" else self.train_limit
            is_test = split_name == "test"
            rows = self.cache.load(
                self.spec.dataset,
                split=split_name,
                subset=self.spec.subset,
                limit=limit,
            )

            for r in rows:
                state = str(r.get(self.spec.question_field, "")).strip()
                choices_raw = r.get(self.spec.choices_field, {})
                ans_key = str(r.get(self.spec.answer_field, "")).strip()

                if not state or not choices_raw:
                    continue

                texts = [str(t).strip() for t in choices_raw.get("text", [])]
                labels = [str(l).strip() for l in choices_raw.get("label", [])]
                if not texts or len(texts) < 2:
                    continue

                target_idx = None
                if ans_key in labels:
                    target_idx = labels.index(ans_key)
                elif ans_key.isdigit():
                    num_idx = int(ans_key)
                    if 0 <= num_idx < len(texts):
                        target_idx = num_idx

                if target_idx is None or target_idx >= len(texts):
                    continue

                tgt_str = texts[target_idx]
                cands_tuple = tuple(texts)
                probs = {c: (1.0 if c == tgt_str else 0.0) for c in cands_tuple}
                case = DecisionCase(
                    state=state,
                    candidates=cands_tuple,
                    probabilities=probs,
                    meta=CaseMeta(
                        category=self.spec.category,
                        split="test" if is_test else "train",
                        source=source_tag,
                    ),
                )
                cases.append(case)

        return cases

__all__ = ["MultiChoiceSpec", "MultiChoiceSource"]
