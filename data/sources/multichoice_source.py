from dataclasses import dataclass
from typing import List, Optional, Sequence
from datasets import load_dataset

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import RAW_DIR, read_raw_jsonl

@dataclass(frozen=True, slots=True)
class MultiChoiceSpec:
    """Core Domain DTO: Declarative specification for multi-choice reasoning datasets."""
    dataset: str
    subset: Optional[str] = None
    question_field: str = "question"
    choices_field: str = "choices"
    answer_field: str = "answerKey"
    category: str = "reasoning"
    instruction: str = "Select the best conclusion:"
    splits: Sequence[str] = ("train", "test")
    train_limit: Optional[int] = None
    test_limit: Optional[int] = 300

MULTICHOICE_REASONING_BENCHMARKS: Sequence[MultiChoiceSpec] = (
    MultiChoiceSpec(
        dataset="allenai/ai2_arc",
        subset="ARC-Challenge",
        question_field="question",
        category="deductive_reasoning",
        instruction="Which option best answers the question?",
    ),
    MultiChoiceSpec(
        dataset="allenai/openbookqa",
        subset="main",
        question_field="question_stem",
        category="causal_reasoning",
        instruction="Which conclusion logically follows?",
        train_limit=1500,
    ),
)

def _load_multichoice_split(spec: MultiChoiceSpec, source_name: str, split: str) -> List[dict]:
    target = RAW_DIR / f"{source_name}_{split}.jsonl"
    def _fetch() -> List[dict]:
        args = [spec.dataset]
        if spec.subset:
            args.append(spec.subset)
        rows = list(load_dataset(*args, split=split))
        limit = spec.test_limit if split == "test" else spec.train_limit
        if limit is not None:
            rows = rows[:limit]
        return rows

    rows = read_raw_jsonl(target, _fetch)
    limit = spec.test_limit if split == "test" else spec.train_limit
    if limit is not None and len(rows) > limit:
        rows = rows[:limit]
    return rows

class MultiChoiceReasoningSource(IDataSource):
    """Universal source adapter for multi-choice reasoning benchmarks (ARC, OpenBookQA)."""

    def __init__(self, spec: Optional[MultiChoiceSpec] = None):
        self.spec = spec or MULTICHOICE_REASONING_BENCHMARKS[0]
        self.name = self.spec.dataset.replace("/", "_")
        if self.spec.subset:
            self.name = f"{self.name}_{self.spec.subset}"

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name in self.spec.splits:
            rows = _load_multichoice_split(self.spec, self.name, split_name)
            tag = "test" if split_name == "test" else "train"
            for r in rows:
                state = r.get(self.spec.question_field, "").strip()
                choices_raw = r.get(self.spec.choices_field, {})
                ans_key = str(r.get(self.spec.answer_field, "")).strip()

                if not state or not choices_raw or not ans_key:
                    continue

                texts = choices_raw.get("text", [])
                labels = choices_raw.get("label", [])
                if not texts or len(texts) < 2:
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

                case = DecisionCase(
                    state=state,
                    questions=[
                        Choice(
                            name="choice",
                            options=texts,
                            instruction=self.spec.instruction,
                            target=texts[target_idx],
                        )
                    ],
                    category=self.spec.category,
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records
