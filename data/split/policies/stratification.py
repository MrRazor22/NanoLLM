from collections import defaultdict
import random
from typing import Iterator, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase

@runtime_checkable
class IStratificationPolicy(Protocol):
    """Sub-Primitive Contract: Yields balanced cases evenly across label classes."""
    def balance(self, cases: Sequence[DecisionCase], rng: random.Random) -> Iterator[DecisionCase]: ...

class RoundRobinStratification(IStratificationPolicy):
    """Fair round-robin class stratification ensuring balanced representation."""

    def balance(self, cases: Sequence[DecisionCase], rng: random.Random) -> Iterator[DecisionCase]:
        if not cases:
            return

        buckets = defaultdict(list)
        for c in cases:
            buckets[c.choice()].append(c)

        for b in buckets.values():
            rng.shuffle(b)

        while any(buckets.values()):
            for b in buckets.values():
                if b:
                    yield b.pop()

__all__ = ["IStratificationPolicy", "RoundRobinStratification"]
