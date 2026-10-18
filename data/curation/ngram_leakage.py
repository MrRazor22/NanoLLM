import re
from typing import Protocol, Sequence, Set, Tuple, Union, runtime_checkable
from data.case import DecisionCase

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
    """Filters out candidate cases that overlap holdouts via exact duplicate, n-gram shingle, Jaccard, or label."""

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

    def filter(
        self,
        candidates: Sequence[DecisionCase],
        holdouts: Sequence[DecisionCase],
    ) -> Sequence[DecisionCase]:
        if not candidates or not holdouts:
            return list(candidates)

        exact_holdouts: Set[Tuple[str, ...]] = set()
        holdout_shingles: Set[Tuple[str, ...]] = set()
        holdout_texts: list[str] = []
        short_holdout_by_first: dict[str, list[str]] = {}
        holdout_targets_by_cat: dict[str, Set[str]] = {}
        holdouts_by_cat: dict[str, list[Tuple[Set[str], int]]] = {}

        disjoint_set = set(self.disjoint_categories)
        exempt_set = set(self.exempt_categories)

        for h in holdouts:
            toks = tokenize_state(h.state)
            cat = h.category or "general"
            if toks:
                exact_holdouts.add(toks)
                if len(toks) >= self.n_gram:
                    holdout_shingles.update(tuple(toks[i : i + self.n_gram]) for i in range(len(toks) - self.n_gram + 1))
                else:
                    holdout_texts.append(" " + " ".join(toks) + " ")
                    if len(toks) <= 8 and toks[0] not in short_holdout_by_first:
                        short_holdout_by_first.setdefault(toks[0], []).append(" " + " ".join(toks) + " ")

            if len(toks) < 30:
                tok_set = set(toks)
                holdouts_by_cat.setdefault(cat, []).append((tok_set, len(tok_set)))

            if cat in disjoint_set and h.option.answer is not None:
                tgt = getattr(h.option.answer, "choice", getattr(h.option.answer, "decision", getattr(h.option.answer, "score", None)))
                if tgt is not None:
                    holdout_targets_by_cat.setdefault(cat, set()).add(str(tgt))

        holdout_blob = (" " + " \n ".join(holdout_texts) + " ") if holdout_texts else ""

        clean: list[DecisionCase] = []
        for c in candidates:
            cat = c.category or "general"
            if cat in exempt_set:
                clean.append(c)
                continue

            if cat in disjoint_set and c.option.answer is not None and cat in holdout_targets_by_cat:
                c_tgt = getattr(c.option.answer, "choice", getattr(c.option.answer, "decision", getattr(c.option.answer, "score", None)))
                if c_tgt is not None and str(c_tgt) in holdout_targets_by_cat[cat]:
                    continue

            toks = tokenize_state(c.state)
            if toks in exact_holdouts:
                continue

            if len(toks) >= self.n_gram:
                if any(tuple(toks[i : i + self.n_gram]) in holdout_shingles for i in range(len(toks) - self.n_gram + 1)):
                    continue

            cand_str = " " + " ".join(toks) + " "
            if len(toks) <= 8:
                cand_set = set(toks)
                if any(p in cand_str for tok in cand_set if tok in short_holdout_by_first for p in short_holdout_by_first[tok]):
                    continue
            elif len(toks) >= self.min_short_tokens and cand_str in holdout_blob:
                continue

            if len(toks) < 30 and cat in holdouts_by_cat:
                cand_set = set(toks)
                cand_len = len(cand_set)
                found_fuzzy = False
                for h_set, h_len in holdouts_by_cat[cat]:
                    common = len(cand_set & h_set)
                    if common / (cand_len + h_len - common) >= self.jaccard_threshold:
                        found_fuzzy = True
                        break
                if found_fuzzy:
                    continue

            clean.append(c)

        return clean
