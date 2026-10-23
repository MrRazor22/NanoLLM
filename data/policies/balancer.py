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
        source_choice_counts = Counter((c.meta.source, c.choice()) for c in cases if c.meta and c.choice())

        def case_slot(c: DecisionCase) -> tuple:
            choice = c.choice()
            source = c.meta.source if c.meta else ""
            candidate_type = c.meta.candidate_type if c.meta else ""
            choice_signature = choice if source_choice_counts.get((source, choice), 0) >= 2 else ""
            instruction_signature = xxhash.xxh64_intdigest(c.instruction.encode()) if c.instruction else 0
            return (
                source,
                candidate_type,
                instruction_signature,
                len(c.candidates),
                c.target_index(),
                choice_signature,
            )

        buckets = defaultdict(list)
        for c in cases:
            buckets[case_slot(c)].append(c)

        groups = list(buckets.values())
        for group in groups:
            rng.shuffle(group)

        target_limit = effective_limit if effective_limit is not None else len(cases)
        selected: list[DecisionCase] = []
        exhausted_count = 0
        num_groups = len(groups)

        while len(selected) < target_limit and exhausted_count < num_groups:
            exhausted_count = 0
            for group in groups:
                if group:
                    selected.append(group.pop())
                    if len(selected) == target_limit:
                        break
                else:
                    exhausted_count += 1

        logger.info(f"Balanced {len(cases)} cases across {len(buckets)} slots to {len(selected)} cases.")
        return tuple(selected)
