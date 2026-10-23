from typing import Iterator, Optional, Sequence

from data.case import CaseMeta, DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

class HellaSwagSource(IDataSource):
    name = "rowan_hellaswag"

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
            for r in self.cache.stream("Rowan/hellaswag", split=split_name, limit=limit):
                ctx_a = str(r.get("ctx_a", "")).strip()
                ctx_b = str(r.get("ctx_b", "")).strip()
                state = f"{ctx_a} {ctx_b}".strip() if ctx_b else ctx_a
                endings = r.get("endings", [])
                label_raw = r.get("label")

                if not state or not endings or label_raw is None or len(endings) < 2:
                    continue

                try:
                    target_idx = int(label_raw)
                except ValueError:
                    continue

                if target_idx < 0 or target_idx >= len(endings):
                    continue

                cleaned_endings = tuple(e.lstrip(", ").strip() for e in endings)
                target_str = cleaned_endings[target_idx]
                probs = {c: (1.0 if c == target_str else 0.0) for c in cleaned_endings}

                yield (
                    DecisionCase(
                        state=state,
                        candidates=cleaned_endings,
                        instruction="",
                        probabilities=probs,
                        meta=CaseMeta(
                            candidate_type="choice",
                            category="adversarial_grounding",
                            split="test" if is_test else "train",
                            source="rowan_hellaswag",
                        ),
                    ),
                )

__all__ = ["HellaSwagSource"]
