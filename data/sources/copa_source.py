from typing import List, Sequence

from data.case import DecisionCase
from data.sources.source import IDataSource
from data.sources.source import load_raw_jsonl

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
                label_raw = r.get("label")

                if not premise or not choice1 or not choice2 or label_raw is None:
                    continue

                try:
                    target_idx = int(label_raw)
                except ValueError:
                    continue

                options = (choice1, choice2)
                if target_idx < 0 or target_idx >= len(options):
                    continue

                target_str = options[target_idx]
                probs = {c: (1.0 if c == target_str else 0.0) for c in options}

                cases.append(
                    DecisionCase(
                        state=premise,
                        candidates=options,
                        probabilities=probs,
                        category="causal_reasoning",
                        split="test" if is_test else "train",
                        source=self.name,
                    )
                )

        return cases

__all__ = ["COPASource"]
