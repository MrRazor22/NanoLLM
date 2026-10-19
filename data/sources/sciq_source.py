from typing import List, Optional, Sequence

from data.case import DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

class SciQSource(IDataSource):
    """Scientific Mechanism & Distractor Elimination Source: scientific reasoning with negative distractors."""

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
            rows = self.cache.load("allenai/sciq", split=split_name, limit=limit)
            for r in rows:
                question = str(r.get("question", "")).strip()
                correct = str(r.get("correct_answer", "")).strip()
                d1 = str(r.get("distractor1", "")).strip()
                d2 = str(r.get("distractor2", "")).strip()
                d3 = str(r.get("distractor3", "")).strip()

                if not question or not correct or not d1 or not d2 or not d3:
                    continue

                opts = [correct, d1, d2, d3]
                if len(set(opts)) < 4:
                    continue

                options = tuple(sorted(opts))
                probs = {c: (1.0 if c == correct else 0.0) for c in options}

                cases.append(
                    DecisionCase(
                        state=question,
                        candidates=options,
                        probabilities=probs,
                        category="scientific_reasoning",
                        split="test" if is_test else "train",
                        source="allenai_sciq",
                    )
                )

        return cases

__all__ = ["SciQSource"]
