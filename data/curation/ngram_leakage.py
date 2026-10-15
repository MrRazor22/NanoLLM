import re
from typing import Protocol, Sequence, Set, Tuple, Union, runtime_checkable
from data.case import DecisionCase
from inference.questions import DecisionState

@runtime_checkable
class ILeakagePolicy(Protocol):
    """Sovereign policy contract: audits candidate training cases against holdout evaluation suites."""
    def find_leaks(
        self,
        candidates: Sequence[DecisionCase],
        holdouts: Sequence[DecisionCase],
    ) -> Sequence[DecisionCase]: ...

class NGramLeakagePolicy(ILeakagePolicy):
    """Input-state and entity-level leakage policy: exact prompt duplicates, n-gram shingling, token Jaccard similarity, and isolated entity holdouts."""

    def __init__(
        self,
        n_gram: int = 10,
        min_short_tokens: int = 4,
        jaccard_threshold: float = 0.70,
        disjoint_categories: Sequence[str] = ("function_calling",),
        exempt_categories: Sequence[str] = ("typed_decisions",),
    ):
        self.n_gram = n_gram
        self.min_short_tokens = min_short_tokens
        self.jaccard_threshold = jaccard_threshold
        self.disjoint_categories = tuple(disjoint_categories)
        self.exempt_categories = tuple(exempt_categories)

    def find_leaks(
        self,
        candidates: Sequence[DecisionCase],
        holdouts: Sequence[DecisionCase],
    ) -> Sequence[DecisionCase]:
        if not candidates or not holdouts:
            return []

        def tokenize(st: Union[DecisionState, str, dict]) -> Tuple[str, ...]:
            txt = " ".join(f"{k} {v}" for k, v in st.items()) if isinstance(st, dict) else str(st or "")
            return tuple(re.findall(r"\w+", txt.lower()))

        exact_holdouts: Set[Tuple[str, ...]] = set()
        holdout_shingles: Set[Tuple[str, ...]] = set()
        holdout_texts: list[str] = []
        short_holdout_by_first: dict[str, list[str]] = {}
        holdout_targets_by_cat: dict[str, Set[str]] = {}
        holdouts_by_cat: dict[str, list[Tuple[Set[str], int]]] = {}

        disjoint_set = set(self.disjoint_categories)
        exempt_set = set(self.exempt_categories)

        for h in holdouts:
            toks = tokenize(h.state)
            cat = h.category or "general"
            if toks:
                exact_holdouts.add(toks)
                if len(toks) >= self.n_gram:
                    holdout_shingles.update(tuple(toks[i : i + self.n_gram]) for i in range(len(toks) - self.n_gram + 1))
                elif len(toks) >= self.min_short_tokens:
                    short_holdout_by_first.setdefault(toks[0], []).append(f" {' '.join(toks)} ")
                holdout_texts.append(" ".join(toks))

                tok_set = set(toks)
                holdouts_by_cat.setdefault(cat, []).append((tok_set, len(tok_set)))

            if cat in disjoint_set and h.gold:
                for tgt in h.gold.values():
                    holdout_targets_by_cat.setdefault(cat, set()).add(str(tgt))

        holdout_blob = (" " + " \n ".join(holdout_texts) + " ") if holdout_texts else ""

        leaked: list[DecisionCase] = []
        for c in candidates:
            cat = c.category or "general"
            if cat in exempt_set:
                continue

            if cat in disjoint_set and c.gold and cat in holdout_targets_by_cat:
                if any(str(tgt) in holdout_targets_by_cat[cat] for tgt in c.gold.values()):
                    leaked.append(c)
                    continue

            toks = tokenize(c.state)
            if not toks:
                continue

            if toks in exact_holdouts:
                leaked.append(c)
                continue

            if len(toks) >= self.n_gram:
                cand_shingles = {tuple(toks[i : i + self.n_gram]) for i in range(len(toks) - self.n_gram + 1)}
                if cand_shingles & holdout_shingles:
                    leaked.append(c)
                    continue
                cand_str = f" {' '.join(toks)} "
                cand_set = set(toks)
                if any(p in cand_str for tok in cand_set if tok in short_holdout_by_first for p in short_holdout_by_first[tok]):
                    leaked.append(c)
                    continue
            elif len(toks) >= self.min_short_tokens:
                cand_str = f" {' '.join(toks)} "
                if cand_str in holdout_blob:
                    leaked.append(c)
                    continue

            if len(toks) < 30 and cat in holdouts_by_cat:
                cand_set = set(toks)
                cand_len = len(cand_set)
                found_fuzzy = False
                for h_set, h_len in holdouts_by_cat[cat]:
                    common = len(cand_set & h_set)
                    if common / (cand_len + h_len - common) >= self.jaccard_threshold:
                        leaked.append(c)
                        found_fuzzy = True
                        break
                if found_fuzzy:
                    continue

        return leaked
