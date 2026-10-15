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

class MultiChoiceReasoningSource(IDataSource):
    """Universal source adapter for multi-choice reasoning benchmarks (ARC, OpenBookQA)."""

    def __init__(self, spec: Optional[MultiChoiceSpec] = None):
        self.spec = spec or MULTICHOICE_REASONING_BENCHMARKS[0]
        self.name = f"{self.spec.dataset.replace('/', '_')}_{self.spec.subset}" if self.spec.subset else self.spec.dataset.replace("/", "_")

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name in self.spec.splits:
            target = RAW_DIR / f"{self.name}_{split_name}.jsonl"
            args = [self.spec.dataset] + ([self.spec.subset] if self.spec.subset else [])
            limit = self.spec.test_limit if split_name == "test" else self.spec.train_limit
            rows = read_raw_jsonl(target, lambda s=split_name: list(load_dataset(*args, split=s)))
            if limit is not None and len(rows) > limit:
                rows = rows[:limit]
            tag = "test" if split_name == "test" else "train"

            for r in rows:
                state = str(r.get(self.spec.question_field, "")).strip()
                choices_raw = r.get(self.spec.choices_field, {})
                ans_key = str(r.get(self.spec.answer_field, "")).strip()
                if not state or not choices_raw or not ans_key:
                    continue
                texts = choices_raw.get("text", [])
                labels = choices_raw.get("label", [])
                if not texts or len(texts) < 2:
                    continue
                target_idx = labels.index(ans_key) if ans_key in labels else (int(ans_key) - 1 if ans_key.isdigit() and 0 < int(ans_key) <= len(texts) else None)
                if target_idx is None or target_idx >= len(texts):
                    continue
                records.append(SourceRecord(
                    sample=DecisionCase(
                        state=state,
                        questions=[Choice(name="choice", options=list(texts), instruction=self.spec.instruction, target=texts[target_idx])],
                        category=self.spec.category,
                    ),
                    tag=tag,
                ))
        return records

__all__ = ["MultiChoiceSpec", "MULTICHOICE_REASONING_BENCHMARKS", "MultiChoiceReasoningSource"]
