import logging
import re
from typing import Protocol, Sequence, runtime_checkable

from data.case import DecisionCase

logger = logging.getLogger(__name__)

def case_words(case: DecisionCase) -> set[str]:
    return set(re.findall(r"\w+", f"{case.state} {case.instruction}".lower()))

def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a and b) else 0.0

@runtime_checkable
class ILeakagePolicy(Protocol):
    def decontaminate(self, cases: Sequence[DecisionCase], holdouts: Sequence[DecisionCase]) -> Sequence[DecisionCase]: ...

class JaccardLeakagePolicy(ILeakagePolicy):
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold

    def decontaminate(self, cases: Sequence[DecisionCase], holdouts: Sequence[DecisionCase]) -> Sequence[DecisionCase]:
        if not cases or not holdouts:
            raise ValueError("Decontamination requires non-empty cases and holdouts.")

        holdout_sets = [words for h in holdouts if (words := case_words(h))]
        clean = [
            c for c in cases
            if (words := case_words(c))
            and not any(jaccard(words, h) >= self.threshold for h in holdout_sets)
        ]
        logger.info(
            "Decontamination: retained %d of %d records against %d holdouts (%d pruned)",
            len(clean),
            len(cases),
            len(holdouts),
            len(cases) - len(clean),
        )
        return clean

__all__ = ["ILeakagePolicy", "JaccardLeakagePolicy"]
