from collections import Counter
import logging
from typing import Protocol, Sequence, runtime_checkable

from data.bucket import CategoryBucket
from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class IBucketPolicy(Protocol):
    def create(self, cases: Sequence[DecisionCase]) -> CategoryBucket: ...

class CaseBucketPolicy(IBucketPolicy):
    def __init__(self, repetition_threshold: int = 2):
        self.repetition_threshold = repetition_threshold

    def create(self, cases: Sequence[DecisionCase]) -> CategoryBucket:
        inst_counts = Counter((c.meta.source, c.instruction) for c in cases if c.instruction)
        opt_counts = Counter((c.meta.source, c.option()) for c in cases if c.option())

        repeated_instructions = frozenset(
            pair for pair, count in inst_counts.items() if count >= self.repetition_threshold
        )
        repeated_options = frozenset(
            pair for pair, count in opt_counts.items() if count >= self.repetition_threshold
        )

        blueprint = CategoryBucket(
            unique_keys=(),
            repeated_instructions=repeated_instructions,
            repeated_options=repeated_options,
        )
        unique_keys = tuple(sorted({blueprint.key(c) for c in cases}, key=lambda k: str(k)))

        return CategoryBucket(
            unique_keys=unique_keys,
            repeated_instructions=repeated_instructions,
            repeated_options=repeated_options,
        )

__all__ = ["CategoryBucket", "IBucketPolicy", "CaseBucketPolicy"]
