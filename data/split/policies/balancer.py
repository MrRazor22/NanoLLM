from collections import Counter, defaultdict
import logging
import random
from typing import Protocol, Sequence, runtime_checkable
import xxhash

from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class ICaseBalancer(Protocol):
    def balance(self, cases: Sequence[DecisionCase], quota: int, rng: random.Random) -> Sequence[DecisionCase]: ...

class RoundRobinCaseBalancer(ICaseBalancer):
    def balance(self, cases: Sequence[DecisionCase], quota: int, rng: random.Random) -> Sequence[DecisionCase]:
        if not cases or quota <= 0:
            raise ValueError("Balancing requires non-empty cases and a positive quota.")

        source_choice_counts = Counter((c.meta.source, c.choice()) for c in cases if c.choice())

        def case_slot(c: DecisionCase) -> tuple:
            choice = c.choice()
            choice_signature = choice if source_choice_counts.get((c.meta.source, choice), 0) >= 2 else ""
            instruction_signature = xxhash.xxh64_intdigest(c.instruction.encode()) if c.instruction else 0
            return (
                c.meta.category,
                c.meta.source,
                c.meta.candidate_type,
                instruction_signature,
                len(c.candidates),
                c.target_index(),
                choice_signature,
            )

        buckets = defaultdict(list)
        for case in cases:
            buckets[case_slot(case)].append(case)

        groups = list(buckets.values())
        for group in groups:
            rng.shuffle(group)

        selected: list[DecisionCase] = []
        while len(selected) < quota and groups:
            for bucket in list(groups):
                if len(selected) >= quota:
                    break
                selected.append(bucket.pop())
                if not bucket:
                    groups.remove(bucket)

        logger.info(
            "Balanced %d cases into %d across %d compound slots",
            len(cases),
            len(selected),
            len(buckets),
        )
        return selected

__all__ = ["ICaseBalancer", "RoundRobinCaseBalancer"]
