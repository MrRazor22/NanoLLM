from typing import List, Sequence

from data.case import ChoiceAnswer, ChoiceOption, DecisionCase
from data.sources.source import IDataSource
from data.sources.huggingface_source import load_raw_jsonl

class COPASource(IDataSource):
    """Choice of Plausible Alternatives (COPA) Source: causal directionality and root cause deduction."""
    name = "superglue_copa"

    def __init__(self, train_limit: int = 500, test_limit: int = 200):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def cases(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for split_name, is_test in [("train", False), ("validation", True)]:
            limit = self.test_limit if is_test else self.train_limit
            rows = load_raw_jsonl(f"copa_{split_name}", "aps/super_glue", split=split_name, subset="copa", limit=limit)
            for r in rows:
                premise = str(r.get("premise", "")).strip()
                choice1 = str(r.get("choice1", "")).strip()
                choice2 = str(r.get("choice2", "")).strip()
                q_type = str(r.get("question", "cause")).strip().lower()
                label_raw = r.get("label")

                if not premise or not choice1 or not choice2 or label_raw is None:
                    continue

                try:
                    target_idx = int(label_raw)
                except ValueError:
                    continue

                options = [choice1, choice2]
                if target_idx < 0 or target_idx >= len(options):
                    continue

                instruction = "What was the cause of the premise?" if q_type == "cause" else "What was the effect of the premise?"
                target_str = options[target_idx]

                case = DecisionCase(
                    state=premise,
                    option=ChoiceOption(
                        options=options,
                        answer=ChoiceAnswer(choice=target_str, confidence=1.0, probabilities={target_str: 1.0}),
                    ),
                    category="causal_reasoning",
                    split="test" if is_test else "train",
                    source=self.name,
                )
                cases.append(case)

        return cases

__all__ = ["COPASource"]
