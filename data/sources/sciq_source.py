from typing import List, Sequence

from data.case import DecisionCase
from data.sources.source import IDataSource
from data.sources.source import load_raw_jsonl

SCIQ_INSTRUCTION = "Which option correctly answers the question?"

class SciQSource(IDataSource):
    """Scientific Mechanism & Distractor Elimination Source: scientific reasoning with negative distractors."""
    name = "allenai_sciq"

    def __init__(self, train_limit: int = 1500, test_limit: int = 200):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def cases(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for split_name, is_test in [("train", False), ("validation", True)]:
            limit = self.test_limit if is_test else self.train_limit
            rows = load_raw_jsonl(f"sciq_{split_name}", "allenai/sciq", split=split_name, limit=limit)
            for r in rows:
                question = str(r.get("question", "")).strip()
                correct = str(r.get("correct_answer", "")).strip()
                d1 = str(r.get("distractor1", "")).strip()
                d2 = str(r.get("distractor2", "")).strip()
                d3 = str(r.get("distractor3", "")).strip()

                if not question or not correct or not d1 or not d2 or not d3:
                    continue

                options = (correct, d1, d2, d3)
                if len(set(options)) < 4:
                    continue

                probs = {c: (1.0 if c == correct else 0.0) for c in options}

                cases.append(
                    DecisionCase(
                        state=question,
                        candidates=options,
                        probabilities=probs,
                        category="scientific_reasoning",
                        split="test" if is_test else "train",
                        source=self.name,
                    )
                )

        return cases

__all__ = ["SciQSource", "SCIQ_INSTRUCTION"]
