import logging
from typing import Iterable, Iterator, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase

logger = logging.getLogger(__name__)

def ngrams(text: str, n: int = 5) -> set[str]:
    """Pure domain transform: decomposes normalized text into character n-grams."""
    cleaned = " ".join(text.lower().split())
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}

def jaccard(a: set[str], b: set[str]) -> float:
    """Computes standard Jaccard intersection-over-union similarity."""
    return len(a & b) / len(a | b) if (a and b) else 0.0

@runtime_checkable
class ILeakagePolicy(Protocol):
    """Sovereign policy contract: filters candidate cases against holdout evaluation suites."""
    def decontaminate(self, candidates: Iterable[DecisionCase], holdouts: Sequence[DecisionCase]) -> Iterator[DecisionCase]: ...

class JaccardLeakagePolicy(ILeakagePolicy):
    """Filters candidate cases sharing high n-gram Jaccard similarity with holdout evaluation cases."""

    def __init__(self, threshold: float = 0.80, n: int = 5):
        self.threshold = threshold
        self.n = n

    def decontaminate(self, candidates: Iterable[DecisionCase], holdouts: Sequence[DecisionCase]) -> Iterator[DecisionCase]:
        if not holdouts:
            yield from candidates
            return

        holdout_grams = [ngrams(h.state, self.n) for h in holdouts if h.state]

        for c in candidates:
            c_grams = ngrams(c.state, self.n)
            if not c_grams:
                continue

            # Reject candidate if it shares >= threshold n-gram Jaccard with any holdout
            if not any(c_grams == h or jaccard(c_grams, h) >= self.threshold for h in holdout_grams):
                yield c

__all__ = ["ILeakagePolicy", "JaccardLeakagePolicy"]
