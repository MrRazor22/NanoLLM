from typing import List, Optional, Sequence

from data.case import DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

class ReClorSource(IDataSource):
    """ReClor Logical Reasoning Source: LSAT multi-constraint argument analysis and fallacy elimination."""

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
            rows = self.cache.load("metaeval/reclor", split=split_name, limit=limit)
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
                cands_tuple = tuple(answers)
                tgt_str = cands_tuple[idx]
                probs = {c: (1.0 if c == tgt_str else 0.0) for c in cands_tuple}

                cases.append(
                    DecisionCase(
                        state=state,
                        candidates=cands_tuple,
                        probabilities=probs,
                        category="logical_reasoning",
                        split="test" if is_test else "train",
                        source="reclor",
                    )
                )

        return cases

__all__ = ["ReClorSource"]
