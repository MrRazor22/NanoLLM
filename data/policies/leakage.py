import logging
from typing import Protocol, Sequence, runtime_checkable

from data.case import DecisionCase

logger = logging.getLogger(__name__)

def jaccard_hashes(a: set[int], b: frozenset[int]) -> float:
    return len(a & b) / len(a | b)

@runtime_checkable
class ILeakagePolicy(Protocol):
    def decontaminate(
        self,
        cases: Sequence[DecisionCase],
        holdouts: Sequence[frozenset[int]],
    ) -> Sequence[DecisionCase]: ...

class JaccardLeakagePolicy(ILeakagePolicy):
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold

    def decontaminate(
        self,
        cases: Sequence[DecisionCase],
        holdouts: Sequence[frozenset[int]],
    ) -> Sequence[DecisionCase]:
        if not cases or not holdouts:
            return cases

        clean = []
        for c in cases:
            h_words = set(c.prompt_hashes())
            if not h_words:
                clean.append(c)
                continue
            w_len = len(h_words)
            if not any(
                jaccard_hashes(h_words, h) >= self.threshold
                for h in holdouts
                if min(w_len, len(h)) / max(w_len, len(h)) >= self.threshold
            ):
                clean.append(c)

        logger.info(
            "Decontamination: retained %d of %d cases against %d holdouts (%d pruned)",
            len(clean),
            len(cases),
            len(holdouts),
            len(cases) - len(clean),
        )
        return clean

__all__ = ["ILeakagePolicy", "JaccardLeakagePolicy", "jaccard_hashes"]
