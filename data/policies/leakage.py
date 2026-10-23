import logging
import re
from typing import Optional, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase

logger = logging.getLogger(__name__)

def case_words(case: DecisionCase) -> set[str]:
    return set(re.findall(r"\w+", f"{case.state} {case.instruction}".lower()))

def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a and b) else 0.0

@runtime_checkable
class ILeakagePolicy(Protocol):
    def decontaminate(self, cases: Sequence[DecisionCase], holdouts: Optional[Sequence[DecisionCase]] = None) -> Sequence[DecisionCase]: ...

class JaccardLeakagePolicy(ILeakagePolicy):
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold

    def decontaminate(self, cases: Sequence[DecisionCase], holdouts: Optional[Sequence[DecisionCase]] = None) -> Sequence[DecisionCase]:
        if not cases:
            return ()

        resolved_holdouts = holdouts if holdouts is not None else [c for c in cases if c.meta.split == "test"]
        if not resolved_holdouts:
            return cases

        holdout_sets = [words for h in resolved_holdouts if (words := case_words(h))]
        pool = [c for c in cases if c.meta.split != "test"]

        clean_pool = [
            c for c in pool
            if (words := case_words(c))
            and not any(jaccard(words, h) >= self.threshold for h in holdout_sets)
        ]

        logger.info(
            "Decontamination: retained %d of %d pool cases against %d holdouts (%d pruned)",
            len(clean_pool),
            len(pool),
            len(resolved_holdouts),
            len(pool) - len(clean_pool),
        )
        return [*resolved_holdouts, *clean_pool]

__all__ = ["ILeakagePolicy", "JaccardLeakagePolicy", "case_words", "jaccard"]
