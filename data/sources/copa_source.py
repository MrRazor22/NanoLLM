from typing import List, Optional, Sequence

from data.case import DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

class COPASource(IDataSource):
    """Choice of Plausible Alternatives (COPA) Source: causal directionality and root cause deduction."""

    def __init__(
        self,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
        cache: Optional[IRawCachePolicy] = None,
    ):
        self.train_limit = train_limit
        self.test_limit = test_limit
        self.cache = cache or HFCache()

    def cases(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for split_name, is_test in [("train", False), ("validation", True)]:
            limit = self.test_limit if is_test else self.train_limit
            rows = self.cache.load("aps/super_glue", split=split_name, subset="copa", limit=limit)
            for r in rows:
                premise = str(r.get("premise", "")).strip()
                question = str(r.get("question", "")).strip()
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
                state = f"{premise}\nWhat was the {question}?" if question else premise

                cases.append(
                    DecisionCase(
                        state=state,
                        candidates=options,
                        probabilities=probs,
                        category="causal_reasoning",
                        split="test" if is_test else "train",
                        source="superglue_copa",
                    )
                )

        return cases

__all__ = ["COPASource"]
