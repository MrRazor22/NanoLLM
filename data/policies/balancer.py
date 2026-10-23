from collections import defaultdict
import logging
import random
from typing import Optional, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase
from data.policies.bucket import CaseBucketPolicy, CategoryBucket, IBucketPolicy

logger = logging.getLogger(__name__)

@runtime_checkable
class ICaseBalancer(Protocol):
    def balance(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]: ...

class RoundRobinCaseBalancer(ICaseBalancer):
    def __init__(self, bucket_policy: Optional[IBucketPolicy] = None, seed: int = 42):
        self.bucket_policy = bucket_policy or CaseBucketPolicy()
        self.seed = seed

    def balance(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]:
        if not cases:
            return ()

        rng = random.Random(self.seed)
        buckets = defaultdict(list)
        for c in cases:
            b = self.bucket_policy.bucket(c)
            buckets[b].append(c)

        groups = list(buckets.values())
        for group in groups:
            rng.shuffle(group)

        selected: list[DecisionCase] = []
        exhausted_count = 0
        num_groups = len(groups)

        while exhausted_count < num_groups:
            exhausted_count = 0
            for group in groups:
                if group:
                    selected.append(group.pop())
                else:
                    exhausted_count += 1

        return tuple(selected)

__all__ = ["ICaseBalancer", "RoundRobinCaseBalancer"]
