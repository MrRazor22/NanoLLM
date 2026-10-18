from typing import List, Sequence

from data.case import ChoiceAnswer, ChoiceOption, DecisionCase
from data.sources.source import IDataSource
from data.sources.huggingface_source import load_raw_jsonl

HELLASWAG_INSTRUCTION = "Which ending logically completes the scenario?"

class HellaSwagSource(IDataSource):
    """Adversarially Filtered Grounding Source: extracts HellaSwag continuation decisions."""
    name = "rowan_hellaswag"

    def __init__(self, train_limit: int = 2000, test_limit: int = 300):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for split_name, is_test in [("train", False), ("validation", True)]:
            limit = self.test_limit if is_test else self.train_limit
            rows = load_raw_jsonl(f"hellaswag_{split_name}", "Rowan/hellaswag", split=split_name, limit=limit)
            for r in rows:
                state = str(r.get("ctx", "")).strip()
                endings = r.get("endings", [])
                label_raw = r.get("label")

                if not state or not endings or label_raw is None:
                    continue

                try:
                    target_idx = int(label_raw)
                except ValueError:
                    continue

                if target_idx < 0 or target_idx >= len(endings):
                    continue

                cleaned_endings = [e.lstrip(", ").strip() for e in endings]
                target_str = cleaned_endings[target_idx]

                case = DecisionCase(
                    state=state,
                    option=ChoiceOption(
                        options=cleaned_endings,
                        answer=ChoiceAnswer(choice=target_str, confidence=1.0, probabilities={target_str: 1.0}),
                    ),
                    category="adversarial_grounding",
                    split="test" if is_test else "train",
                    source=self.name,
                )
                cases.append(case)

        return cases

__all__ = ["HellaSwagSource", "HELLASWAG_INSTRUCTION"]
