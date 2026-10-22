from typing import List, Optional, Sequence

from data.case import CaseMeta, DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

NLI_OPTIONS = ("Entailment", "Neutral", "Contradiction")

class ANLISource(IDataSource):
    """Adversarial NLI (ANLI) dataset source: extracts human-crafted counterfactuals and refutations."""

    def __init__(
        self,
        rounds: Sequence[str] = ("r1", "r2", "r3"),
        train_limit_per_round: Optional[int] = None,
        test_limit_per_round: Optional[int] = None,
        cache: Optional[IRawCachePolicy] = None,
    ):
        self.rounds = list(rounds)
        self.train_limit = train_limit_per_round
        self.test_limit = test_limit_per_round
        self.cache = cache or HFCache()

    def cases(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for r_name in self.rounds:
            for split_type in ("train", "test"):
                limit = self.test_limit if split_type == "test" else self.train_limit
                hf_split = f"{split_type}_{r_name}"
                rows = self.cache.load("facebook/anli", split=hf_split, limit=limit)
                is_test = split_type == "test"
                for r in rows:
                    premise = str(r.get("premise", "")).strip()
                    hypothesis = str(r.get("hypothesis", "")).strip()
                    label = r.get("label")

                    if not premise or not hypothesis or label not in (0, 1, 2):
                        continue

                    lbl_str = NLI_OPTIONS[int(label)]
                    state = f"Premise: {premise}\nHypothesis: {hypothesis}"
                    probs = {c: (1.0 if c == lbl_str else 0.0) for c in NLI_OPTIONS}
                    cases.append(
                        DecisionCase(
                            state=state,
                            candidates=NLI_OPTIONS,
                            probabilities=probs,
                            meta=CaseMeta(
                                category="adversarial_reasoning",
                                split="test" if is_test else "train",
                                source="facebook_anli",
                            ),
                        )
                    )

        return cases

__all__ = ["ANLISource", "NLI_OPTIONS"]
