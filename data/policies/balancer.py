from collections import defaultdict
import logging
import random
from typing import Optional, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase
from data.policies.census import CategoryBucket

logger = logging.getLogger(__name__)

@runtime_checkable
class ICaseBalancer(Protocol):
    def balance(
        self,
        cases: Sequence[DecisionCase],
        category_bucket: CategoryBucket,
        limit: Optional[int] = None,
    ) -> Sequence[DecisionCase]: ...

class RoundRobinCaseBalancer(ICaseBalancer):
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def balance(
        self,
        cases: Sequence[DecisionCase],
        category_bucket: CategoryBucket,
        limit: Optional[int] = None,
    ) -> Sequence[DecisionCase]:
        if not cases:
            return ()

        known_instructions = set(category_bucket.instructions)
        known_candidates = set(category_bucket.candidates)

        buckets: dict[tuple, list[DecisionCase]] = defaultdict(list)
        for case in cases:
            inst_sig = case.instruction_signature()
            cand_sig = case.option_signature()
            instruction_signature = inst_sig if (not known_instructions or inst_sig in known_instructions) else 0
            candidate_signature = cand_sig if (not known_candidates or cand_sig in known_candidates) else 0

            key = (
                case.meta.source,
                case.meta.candidate_type,
                instruction_signature,
                len(case.candidates),
                case.target_index(),
                candidate_signature,
            )
            buckets[key].append(case)

        bucket_queues = list(buckets.values())
        for queue in bucket_queues:
            self.rng.shuffle(queue)
        self.rng.shuffle(bucket_queues)

        selected: list[DecisionCase] = []
        exhausted_count = 0
        num_buckets = len(bucket_queues)

        while exhausted_count < num_buckets:
            exhausted_count = 0
            for queue in bucket_queues:
                if limit is not None and len(selected) >= limit:
                    break
                if queue:
                    selected.append(queue.pop())
                else:
                    exhausted_count += 1
            if limit is not None and len(selected) >= limit:
                break

        logger.debug(f"Balanced {len(cases)} cases across {len(buckets)} compound slots into {len(selected)} cases.")
        return tuple(selected)

__all__ = ["ICaseBalancer", "RoundRobinCaseBalancer"]
