import logging
import re
from typing import Protocol, Sequence, Set, Tuple, runtime_checkable
from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class ILeakagePolicy(Protocol):
    """Sovereign policy contract: filters candidate cases against holdout evaluation suites."""
    def filter(
        self,
        candidates: Sequence[DecisionCase],
        holdouts: Sequence[DecisionCase],
    ) -> Sequence[DecisionCase]: ...

def tokenize_state(st: str) -> Tuple[str, ...]:
    """Pure domain transform: decomposes state text into lowercase lexical tokens."""
    return tuple(re.findall(r"\w+", st.lower()))

class NGramLeakagePolicy(ILeakagePolicy):
    """Filters candidate cases that duplicate or share n-gram shingles with holdout cases."""

    def __init__(self, n_gram: int = 10):
        self.n_gram = n_gram

    def filter(
        self,
        candidates: Sequence[DecisionCase],
        holdouts: Sequence[DecisionCase],
    ) -> Sequence[DecisionCase]:
        if not candidates or not holdouts:
            return list(candidates)
        logger.info("Auditing %d candidates against %d holdouts (%d-gram)", len(candidates), len(holdouts), self.n_gram)

        exact_holdouts: Set[Tuple[str, ...]] = set()
        holdout_shingles: Set[Tuple[str, ...]] = set()

        for h in holdouts:
            toks = tokenize_state(h.state)
            if toks:
                exact_holdouts.add(toks)
                if len(toks) >= self.n_gram:
                    holdout_shingles.update(toks[i : i + self.n_gram] for i in range(len(toks) - self.n_gram + 1))

        clean: list[DecisionCase] = []
        for c in candidates:
            toks = tokenize_state(c.state)
            if toks in exact_holdouts:
                continue
            if len(toks) >= self.n_gram and any(toks[i : i + self.n_gram] in holdout_shingles for i in range(len(toks) - self.n_gram + 1)):
                continue
            clean.append(c)

        return clean

__all__ = ["ILeakagePolicy", "NGramLeakagePolicy"]
