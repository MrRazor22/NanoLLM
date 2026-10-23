from typing import List, Optional, Sequence

from data.case import CaseMeta, DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

class ARTSource(IDataSource):
    """Abductive Reasoning in Natural Language (aNLI / ART) Source: deduces latent unstated hypotheses."""

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
            rows = self.cache.load("allenai/art", split=split_name, limit=limit)
            for r in rows:
                obs1 = str(r.get("observation_1", "")).strip()
                obs2 = str(r.get("observation_2", "")).strip()
                hyp1 = str(r.get("hypothesis_1", "")).strip()
                hyp2 = str(r.get("hypothesis_2", "")).strip()
                label_raw = r.get("label")

                if not obs1 or not obs2 or not hyp1 or not hyp2 or label_raw is None:
                    continue

                try:
                    raw_idx = int(label_raw)
                    target_idx = (raw_idx - 1) if raw_idx in (1, 2) else raw_idx
                except ValueError:
                    continue

                options = (hyp1, hyp2)
                if target_idx < 0 or target_idx >= len(options):
                    continue

                state = f"Observation 1: {obs1}\nObservation 2: {obs2}"
                target_str = options[target_idx]
                probs = {c: (1.0 if c == target_str else 0.0) for c in options}

                cases.append(
                    DecisionCase(
                        state=state,
                        candidates=options,
                        probabilities=probs,
                        meta=CaseMeta(
                                option_type='choice',
                            category="abductive_reasoning",
                            split="test" if is_test else "train",
                            source="allenai_art",
                        ),
                    )
                )

        return cases

__all__ = ["ARTSource"]
