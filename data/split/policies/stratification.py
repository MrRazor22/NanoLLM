from collections import defaultdict
import logging
import random
from typing import Protocol, Sequence, runtime_checkable

from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class IStratificationPolicy(Protocol):
    def balance(self, cases: Sequence[DecisionCase], quota: int, rng: random.Random) -> Sequence[DecisionCase]: ...

class RoundRobinStratification(IStratificationPolicy):
    def balance(self, cases: Sequence[DecisionCase], quota: int, rng: random.Random) -> Sequence[DecisionCase]:
        if not cases or quota <= 0:
            raise ValueError("Stratification requires non-empty cases and a positive quota.")

        buckets = defaultdict(list)
        for case in cases:
            buckets[(len(case.candidates), case.target_index())].append(case)

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
            "Stratified %d cases into %d balanced across %d slots (%s)",
            len(cases),
            len(selected),
            len(buckets),
            ", ".join(f"{k[0]}c-idx{k[1]}:{len([c for c in selected if (len(c.candidates), c.target_index()) == k])}" for k in sorted(buckets.keys())),
        )
        return selected

__all__ = ["IStratificationPolicy", "RoundRobinStratification"]
