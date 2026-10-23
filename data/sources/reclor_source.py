from typing import Iterator, Optional, Sequence

from data.case import CaseMeta, DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

class ReClorSource(IDataSource):
    name = "reclor"

    def __init__(
        self,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
        cache: Optional[IRawCachePolicy] = None,
    ):
        self.train_limit = train_limit
        self.test_limit = test_limit
        self.cache = cache or HFCache()

    def stream(self) -> Iterator[Sequence[DecisionCase]]:
        for split_name, is_test in [("train", False), ("validation", True)]:
            limit = self.test_limit if is_test else self.train_limit
            for r in self.cache.stream("metaeval/reclor", split=split_name, limit=limit):
                context = str(r.get("context", "")).strip()
                question = str(r.get("question", "")).strip()
                answers = [str(a).strip() for a in r.get("answers", [])]
                label = r.get("label")

                if not context or not answers or label is None or len(answers) < 2:
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

                yield (
                    DecisionCase(
                        state=state,
                        candidates=cands_tuple,
                        instruction="",
                        probabilities=probs,
                        meta=CaseMeta(
                            candidate_type="choice",
                            category="logical_reasoning",
                            split="test" if is_test else "train",
                            source="reclor",
                        ),
                    ),
                )

__all__ = ["ReClorSource"]
