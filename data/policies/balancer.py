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
        by_partition: dict[tuple[str, str], list[DecisionCase]] = {}
        for c in cases:
            cat = c.meta.category if c.meta else ""
            split = c.meta.split if c.meta and c.meta.split else ""
            by_partition.setdefault((cat, split), []).append(c)

        balanced: list[DecisionCase] = []
        for (cat, split), part_cases in by_partition.items():
            source_choice_counts = Counter((c.meta.source, c.choice()) for c in part_cases if c.choice())

            def case_slot(c: DecisionCase) -> tuple:
                choice = c.choice()
                choice_signature = choice if source_choice_counts.get((c.meta.source, choice), 0) >= 2 else ""
                instruction_signature = xxhash.xxh64_intdigest(c.instruction.encode()) if c.instruction else 0
                return (
                    c.meta.category if c.meta else "",
                    c.meta.source if c.meta else "",
                    c.meta.candidate_type if c.meta else "",
                    instruction_signature,
                    len(c.candidates),
                    c.target_index(),
                    choice_signature,
                )

            buckets = defaultdict(list)
            for c in part_cases:
                buckets[case_slot(c)].append(c)

            groups = list(buckets.values())
            for group in groups:
                rng.shuffle(group)

            target_limit = effective_limit if effective_limit is not None else len(part_cases)
            part_selected: list[DecisionCase] = []
            while len(part_selected) < target_limit and groups:
                for bucket in list(groups):
                    if len(part_selected) >= target_limit:
                        break
                    part_selected.append(bucket.pop())
                    if not bucket:
                        groups.remove(bucket)

            balanced.extend(part_selected)

        logger.info(
            "Balanced %d cases into %d across %d partitions",
            len(cases),
            len(balanced),
            len(by_partition),
        )
        return balanced

__all__ = ["ICaseBalancer", "RoundRobinCaseBalancer"]
