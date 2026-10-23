from collections import defaultdict
import logging
import random
from typing import Optional, Protocol, Sequence, runtime_checkable
import xxhash

from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class ICaseBalancer(Protocol):
    def bucket(self, case: DecisionCase) -> tuple: ...
    def balance(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]: ...

class RoundRobinCaseBalancer(ICaseBalancer):
    def __init__(self, seed: int = 42):
        self.seed = seed

    def bucket(self, case: DecisionCase) -> tuple:
        choice = case.choice()
        source = case.meta.source
        candidate_type = case.meta.candidate_type
        instruction_signature = xxhash.xxh64_intdigest(case.instruction.encode()) if case.instruction else 0
        return (
            source,
            candidate_type,
            instruction_signature,
            len(case.candidates),
            case.target_index(),
            choice,
        )

    def balance(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]:
        if not cases:
            return ()

        rng = random.Random(self.seed)
        buckets = defaultdict(list)
        for c in cases:
            buckets[self.bucket(c)].append(c)

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
