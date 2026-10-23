from collections import Counter, defaultdict
import logging
import random
from typing import Optional, Protocol, Sequence, runtime_checkable
import xxhash

from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class ICaseBalancer(Protocol):
    def balance(self, cases: Sequence[DecisionCase], limit: Optional[int] = None) -> Sequence[DecisionCase]: ...

class RoundRobinCaseBalancer(ICaseBalancer):
    def __init__(self, limit: Optional[int] = None, seed: int = 42):
        self.limit = limit
        self.seed = seed

    def balance(self, cases: Sequence[DecisionCase], limit: Optional[int] = None) -> Sequence[DecisionCase]:
        if not cases:
            return ()

        effective_limit = limit if limit is not None else self.limit
        if effective_limit is not None and effective_limit <= 0:
            return ()

        rng = random.Random(self.seed)
        by_category: dict[str, list[DecisionCase]] = {}
        for c in cases:
            cat = c.meta.category if c.meta else ""
            by_category.setdefault(cat, []).append(c)

        balanced: list[DecisionCase] = []
        for cat, cat_cases in by_category.items():
            source_choice_counts = Counter((c.meta.source, c.choice()) for c in cat_cases if c.choice())

            def case_slot(c: DecisionCase) -> tuple:
                choice = c.choice()
                choice_signature = choice if source_choice_counts.get((c.meta.source, choice), 0) >= 2 else ""
                instruction_signature = xxhash.xxh64_intdigest(c.instruction.encode()) if c.instruction else 0
                return (
                    c.meta.source if c.meta else "",
                    c.meta.candidate_type if c.meta else "",
                    instruction_signature,
                    len(c.candidates),
                    c.target_index(),
                    choice_signature,
                )

            buckets = defaultdict(list)
            for c in cat_cases:
                buckets[case_slot(c)].append(c)

            groups = list(buckets.values())
            for group in groups:
                rng.shuffle(group)

            target_limit = effective_limit if effective_limit is not None else len(cat_cases)
            cat_selected: list[DecisionCase] = []
            while len(cat_selected) < target_limit and groups:
                for bucket in list(groups):
                    if len(cat_selected) >= target_limit:
                        break
                    cat_selected.append(bucket.pop())
                    if not bucket:
                        groups.remove(bucket)

            balanced.extend(cat_selected)

        logger.info(
            "Balanced %d cases into %d across %d categories",
            len(cases),
            len(balanced),
            len(by_category),
        )
        return balanced

__all__ = ["ICaseBalancer", "RoundRobinCaseBalancer"]
