from typing import List, Optional, Sequence

from data.case import DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

class MedQASource(IDataSource):
    """Differential Elimination Source: extracts multi-variable clinical elimination decisions."""

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
        for split_name in ("train", "test"):
            limit = self.test_limit if split_name == "test" else self.train_limit
            rows = self.cache.load("GBaker/MedQA-USMLE-4-options", split=split_name, limit=limit)
            is_test = split_name == "test"
            for r in rows:
                state = str(r.get("question", "")).strip()
                opts_dict = r.get("options", {})
                answer_str = str(r.get("answer", "")).strip()

                if not state or not opts_dict or not answer_str:
                    continue

                options_tuple = tuple(opts_dict.values())
                if len(options_tuple) < 2 or answer_str not in options_tuple:
                    continue

                probs = {c: (1.0 if c == answer_str else 0.0) for c in options_tuple}

                cases.append(
                    DecisionCase(
                        state=state,
                        candidates=options_tuple,
                        probabilities=probs,
                        category="differential_elimination",
                        split="test" if is_test else "train",
                        source="gbaker_medqa",
                    )
                )

        return cases

__all__ = ["MedQASource"]
