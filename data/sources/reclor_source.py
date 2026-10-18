from typing import List, Sequence

from data.case import ChoiceAnswer, ChoiceOption, DecisionCase
from data.sources.source import IDataSource
from data.sources.source import load_raw_jsonl

class ReClorSource(IDataSource):
    """ReClor Logical Reasoning Source: LSAT multi-constraint argument analysis and fallacy elimination."""
    name = "reclor"

    def __init__(self, train_limit: int = 2000, test_limit: int = 400):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def cases(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for split_name, is_test in [("train", False), ("validation", True)]:
            limit = self.test_limit if is_test else self.train_limit
            rows = load_raw_jsonl(f"reclor_{split_name}", "metaeval/reclor", split=split_name, limit=limit)
            for r in rows:
                context = str(r.get("context", "")).strip()
                question = str(r.get("question", "")).strip()
                answers = r.get("answers", [])
                label = r.get("label")

                if not context or not answers or len(answers) < 2 or label is None:
                    continue

                try:
                    idx = int(label)
                except (ValueError, TypeError):
                    continue

                if idx < 0 or idx >= len(answers):
                    continue

                state = f"Context: {context}\nQuestion: {question}" if question else context
                tgt_str = answers[idx]
                case = DecisionCase(
                    state=state,
                    option=ChoiceOption(
                        options=answers,
                        answer=ChoiceAnswer(choice=tgt_str, confidence=1.0, probabilities={tgt_str: 1.0}),
                    ),
                    category="logical_reasoning",
                    split="test" if is_test else "train",
                    source=self.name,
                )
                cases.append(case)

        return cases

__all__ = ["ReClorSource"]
