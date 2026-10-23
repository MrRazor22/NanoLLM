from collections import defaultdict
import logging
import random
from typing import Protocol, Sequence, runtime_checkable

from data.bucket import CategoryBucket
from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class ICaseBalancer(Protocol):
    def balance(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]: ...

class RoundRobinCaseBalancer(ICaseBalancer):
    def __init__(self, bucket: CategoryBucket, seed: int = 42):
        self.bucket = bucket
        self.seed = seed
        self.rng = random.Random(seed)
        self.known_keys = list(bucket.unique_keys)
        self.rng.shuffle(self.known_keys)

    def balance(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]:
        if not cases:
            return ()

        cases_by_key = defaultdict(list)
        for case in cases:
            cases_by_key[self.bucket.key(case)].append(case)

        for group in cases_by_key.values():
            self.rng.shuffle(group)

        active_keys = [key for key in self.known_keys if key in cases_by_key and cases_by_key[key]]
        known = set(active_keys)
        active_keys.extend(key for key in cases_by_key if key not in known and cases_by_key[key])

        selected: list[DecisionCase] = []
        exhausted_count = 0
        num_keys = len(active_keys)

        while exhausted_count < num_keys:
            exhausted_count = 0
            for key in active_keys:
                items = cases_by_key[key]
                if items:
                    selected.append(items.pop())
                else:
                    exhausted_count += 1

        return tuple(selected)

__all__ = ["ICaseBalancer", "RoundRobinCaseBalancer"]
