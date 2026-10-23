from collections import Counter, defaultdict
import hashlib
import logging
import random
from typing import Protocol, Sequence, runtime_checkable

from data.case import DecisionCase

logger = logging.getLogger(__name__)

def simhash16(text: str) -> int:
    words = text.lower().split()
    if not words:
        return 0
    vector = [0] * 16
    for word in words:
        digest = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=2).digest(), "big")
        for i in range(16):
            vector[i] += 1 if (digest & (1 << i)) else -1
    signature = 0
    for i in range(16):
        if vector[i] > 0:
            signature |= 1 << i
    return signature

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
            return (
                c.meta.category,
                c.meta.source,
                c.meta.option_type,
                simhash16(c.instruction),
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

__all__ = ["ICaseBalancer", "RoundRobinCaseBalancer", "simhash16"]
