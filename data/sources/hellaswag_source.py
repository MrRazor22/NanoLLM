from typing import List, Optional, Sequence

from data.case import DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

HELLASWAG_INSTRUCTION = "Which ending logically completes the scenario?"

class HellaSwagSource(IDataSource):
    """Adversarially Filtered Grounding Source: extracts HellaSwag continuation decisions."""
    name = "rowan_hellaswag"

    def __init__(
        self,
        train_limit: int = 2000,
        test_limit: int = 300,
        cache: Optional[IRawCachePolicy] = None,
    ):
        self.train_limit = train_limit
        self.test_limit = test_limit
        self.cache = cache or HFCache()

    def cases(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for split_name, is_test in [("train", False), ("validation", True)]:
            limit = self.test_limit if is_test else self.train_limit
            rows = self.cache.load("Rowan/hellaswag", split=split_name, limit=limit)
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

                cleaned_endings = tuple(e.lstrip(", ").strip() for e in endings)
                target_str = cleaned_endings[target_idx]
                probs = {c: (1.0 if c == target_str else 0.0) for c in cleaned_endings}

                cases.append(
                    DecisionCase(
                        state=state,
                        candidates=cleaned_endings,
                        probabilities=probs,
                        category="adversarial_grounding",
                        split="test" if is_test else "train",
                        source=self.name,
                    )
                )

        return cases

__all__ = ["HellaSwagSource", "HELLASWAG_INSTRUCTION"]
