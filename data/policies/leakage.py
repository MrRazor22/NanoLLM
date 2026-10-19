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

def tokenize_words(st: str) -> set[str]:
    """Pure domain transform: decomposes state text into a set of lowercase lexical tokens."""
    return set(re.findall(r"\w+", st.lower()))

class NGramLeakagePolicy(ILeakagePolicy):
    """Filters candidate cases that duplicate or share high lexical similarity with holdout cases within the same category."""

    def __init__(self, jaccard_threshold: float = 0.80):
        self.jaccard_threshold = jaccard_threshold

    def filter(
        self,
        candidates: Sequence[DecisionCase],
        holdouts: Sequence[DecisionCase],
    ) -> Sequence[DecisionCase]:
        if not candidates or not holdouts:
            return list(candidates)

        holdouts_by_cat: dict[str, list[set[str]]] = {}
        for h in holdouts:
            toks = tokenize_words(h.state)
            if toks:
                holdouts_by_cat.setdefault(h.category, []).append(toks)

        clean: list[DecisionCase] = []
        for c in candidates:
            c_toks = tokenize_words(c.state)
            if not c_toks:
                continue

            h_list = holdouts_by_cat.get(c.category)
            if h_list:
                is_leak = False
                for h_toks in h_list:
                    if c_toks == h_toks:
                        is_leak = True
                        break
                    inter = len(c_toks & h_toks)
                    if inter / len(c_toks | h_toks) >= self.jaccard_threshold:
                        is_leak = True
                        break
                if is_leak:
                    continue

            clean.append(c)

        logger.info(
            "Audited %d candidates vs %d holdouts (Jaccard >= %.0f%%) -> kept %d, pruned %d",
            len(candidates),
            len(holdouts),
            self.jaccard_threshold * 100,
            len(clean),
            len(candidates) - len(clean),
        )
        return clean

__all__ = ["ILeakagePolicy", "NGramLeakagePolicy"]
