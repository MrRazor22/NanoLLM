from dataclasses import dataclass
from typing import List, Optional, Sequence
from data.case import DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

@dataclass(frozen=True, slots=True)
class MultiChoiceSpec:
    """Core Domain DTO: Declarative specification for multi-choice reasoning datasets."""
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
    """Universal source adapter for multi-choice decision benchmarks (ARC, OpenBookQA)."""

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
            rows = self.cache.load(
                self.spec.dataset,
                split=split_name,
                subset=self.spec.subset,
                limit=limit,
            )
            is_test = split_name == "test"
            for r in rows:
                state = str(r.get(self.spec.question_field) or "").strip()
                choices_raw = r.get(self.spec.choices_field) or {}
                ans_key = str(r.get(self.spec.answer_field) or "").strip()

                if not state or not isinstance(choices_raw, dict) or not ans_key:
                    continue

                texts = [str(t).strip() for t in choices_raw.get("text", []) if str(t).strip()]
                labels = [str(lbl).strip() for lbl in choices_raw.get("label", [])]
                if len(texts) < 2:
                    continue

                target_idx: Optional[int] = None
                if ans_key in labels:
                    target_idx = labels.index(ans_key)
                elif ans_key.isdigit():
                    num_idx = int(ans_key) - 1
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
                    category=self.spec.category,
                    split="test" if is_test else "train",
                    source=source_tag,
                )
                cases.append(case)

        return cases

__all__ = ["MultiChoiceSpec", "MultiChoiceSource"]
