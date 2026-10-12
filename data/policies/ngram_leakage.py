import re
from typing import Protocol, Sequence, Set, Tuple, runtime_checkable
from data.case import DecisionCase

@runtime_checkable
class ILeakagePolicy(Protocol):
    """Sovereign policy contract: audits candidate training cases against holdout evaluation suites."""
    @property
    def signature(self) -> str: ...
    def find_leaks(
        self,
        candidates: Sequence[DecisionCase],
        holdouts: Sequence[DecisionCase],
    ) -> Sequence[DecisionCase]: ...

class NGramLeakagePolicy(ILeakagePolicy):
    """Input-state leakage policy: exact prompt duplicates, short-to-long sub-sequences, and n-gram shingling."""

    def __init__(self, n_gram: int = 13, min_short_tokens: int = 4):
        self.n_gram = n_gram
        self.min_short_tokens = min_short_tokens

    @property
    def signature(self) -> str:
        import hashlib
        import json
        payload = {
            "type": "NGramLeakagePolicy",
            "n_gram": self.n_gram,
            "min_short_tokens": self.min_short_tokens,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]

    def state_tokens(self, state: object) -> Tuple[str, ...]:
        if isinstance(state, dict):
            text = " ".join(f"{k} {v}" for k, v in state.items())
        else:
            text = str(state or "")
        return tuple(re.findall(r"\w+", text.lower()))

    def extract_shingles(self, tokens: Sequence[str]) -> Set[Tuple[str, ...]]:
        if len(tokens) < self.n_gram:
            return set()
        return {tuple(tokens[i : i + self.n_gram]) for i in range(len(tokens) - self.n_gram + 1)}

    def find_leaks(
        self,
        candidates: Sequence[DecisionCase],
        holdouts: Sequence[DecisionCase],
    ) -> Sequence[DecisionCase]:
        if not candidates or not holdouts:
            return []

        exact_holdouts: Set[Tuple[str, ...]] = set()
        holdout_shingles: Set[Tuple[str, ...]] = set()
        holdout_texts: list[str] = []
        short_holdout_by_first: dict[str, list[str]] = {}

        for h in holdouts:
            toks = self.state_tokens(h.state)
            if toks:
                exact_holdouts.add(toks)
                if len(toks) >= self.n_gram:
                    holdout_shingles.update(self.extract_shingles(toks))
                elif len(toks) >= self.min_short_tokens:
                    short_holdout_by_first.setdefault(toks[0], []).append(f" {' '.join(toks)} ")
                holdout_texts.append(" ".join(toks))

        holdout_blob = (" " + " \n ".join(holdout_texts) + " ") if holdout_texts else ""

        leaked: list[DecisionCase] = []
        for c in candidates:
            toks = self.state_tokens(c.state)
            if not toks:
                continue

            # 1. Exact input/state duplicate (catches 1:1 duplicates regardless of label)
            if toks in exact_holdouts:
                leaked.append(c)
                continue

            # 2. Continuous n-gram state overlap or embedded short holdout
            if len(toks) >= self.n_gram:
                if self.extract_shingles(toks) & holdout_shingles:
                    leaked.append(c)
                    continue
                cand_str = f" {' '.join(toks)} "
                cand_set = set(toks)
                if any(p in cand_str for tok in cand_set if tok in short_holdout_by_first for p in short_holdout_by_first[tok]):
                    leaked.append(c)
                    continue
            elif len(toks) >= self.min_short_tokens:
                # 3. Short candidate query appearing as contiguous sub-sequence in holdout state
                cand_str = f" {' '.join(toks)} "
                if cand_str in holdout_blob:
                    leaked.append(c)
                    continue

        return leaked

__all__ = ["ILeakagePolicy", "NGramLeakagePolicy"]
