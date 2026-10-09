import re
from typing import Protocol, Sequence, Set, Tuple, runtime_checkable
from inference.questions import DecisionSample

@runtime_checkable
class ILeakagePolicy(Protocol):
    """Sovereign policy contract: audits candidate training samples against holdout evaluation suites."""

    def find_leaks(
        self,
        candidates: Sequence[DecisionSample],
        holdouts: Sequence[DecisionSample],
    ) -> Sequence[DecisionSample]: ...

class NGramLeakagePolicy(ILeakagePolicy):
    """13-gram token shingling policy: flags continuous n-gram overlaps against holdouts."""

    def __init__(self, n_gram: int = 13):
        self.n_gram = n_gram

    def extract_shingles(self, text: str) -> Set[Tuple[str, ...]]:
        tokens = re.findall(r"\w+", str(text or "").lower())
        if not tokens:
            return set()
        if len(tokens) < self.n_gram:
            return {tuple(tokens)}
        return {tuple(tokens[i : i + self.n_gram]) for i in range(len(tokens) - self.n_gram + 1)}

    def find_leaks(
        self,
        candidates: Sequence[DecisionSample],
        holdouts: Sequence[DecisionSample],
    ) -> Sequence[DecisionSample]:
        holdout_shingles: Set[Tuple[str, ...]] = set()
        for h in holdouts:
            if h.state:
                holdout_shingles.update(self.extract_shingles(h.state))

        if not holdout_shingles:
            return []

        return [c for c in candidates if c.state and (self.extract_shingles(c.state) & holdout_shingles)]

__all__ = ["ILeakagePolicy", "NGramLeakagePolicy"]
