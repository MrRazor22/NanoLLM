from dataclasses import dataclass
from typing import List, Optional, Sequence
from data.case import ChoiceAnswer, ChoiceOption, DecisionCase
from data.sources.source import IDataSource
from data.sources.source import load_raw_jsonl

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
    MultiChoiceSpec(
        dataset="tau/commonsense_qa",
        question_field="question",
        category="commonsense_reasoning",
        instruction="Which option best answers the question?",
        train_limit=2000,
    ),
    MultiChoiceSpec(
        dataset="allenai/qasc",
        question_field="question",
        category="multihop_reasoning",
        instruction="Which option best completes the reasoning chain?",
        train_limit=2000,
    ),
)

class MultiChoiceReasoningSource(IDataSource):
    """Universal source adapter for multi-choice reasoning benchmarks (ARC, OpenBookQA)."""

    def __init__(self, spec: Optional[MultiChoiceSpec] = None):
        self.spec = spec or MULTICHOICE_REASONING_BENCHMARKS[0]
        self.name = self.spec.dataset.replace("/", "_")
        if self.spec.subset:
            self.name = f"{self.name}_{self.spec.subset}"

    def cases(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for split_name in self.spec.splits:
            limit = self.spec.test_limit if split_name == "test" else self.spec.train_limit
            rows = load_raw_jsonl(
                f"{self.name}_{split_name}",
                self.spec.dataset,
                split=split_name,
                subset=self.spec.subset,
                limit=limit,
            )
            is_test = split_name == "test"
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

                tgt_str = texts[target_idx]
                case = DecisionCase(
                    state=state,
                    option=ChoiceOption(
                        options=texts,
                        answer=ChoiceAnswer(choice=tgt_str, confidence=1.0, probabilities={tgt_str: 1.0}),
                    ),
                    category=self.spec.category,
                    split="test" if is_test else "train",
                    source=self.name,
                )
                cases.append(case)

        return cases
