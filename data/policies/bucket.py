from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import Protocol, Sequence, runtime_checkable
import xxhash

from data.case import DecisionCase

logger = logging.getLogger(__name__)

def text_signature(text: str) -> int:
    return xxhash.xxh64_intdigest(text.encode()) if text else 0

@dataclass(frozen=True, slots=True)
class CategoryBucket:
    category: str
    sources: tuple[str, ...]
    candidate_types: tuple[str, ...]
    instructions: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    target_indices: tuple[int, ...]
    candidates: tuple[int, ...]

@runtime_checkable
class IBucketPolicy(Protocol):
    def create(self, cases: Sequence[DecisionCase]) -> Sequence[CategoryBucket]: ...

class CaseBucketPolicy(IBucketPolicy):
    def create(self, cases: Sequence[DecisionCase]) -> Sequence[CategoryBucket]:
        if not cases:
            return ()

        by_category: dict[str, list[DecisionCase]] = defaultdict(list)
        for c in cases:
            by_category[c.meta.category].append(c)

        result: list[CategoryBucket] = []
        for category, pool in sorted(by_category.items(), key=lambda item: item[0]):
            sources = tuple(sorted({c.meta.source for c in pool}))
            candidate_types = tuple(sorted({c.meta.candidate_type for c in pool}))
            instructions = tuple(sorted({text_signature(c.instruction) for c in pool if c.instruction}))
            candidate_counts = tuple(sorted({len(c.candidates) for c in pool}))
            target_indices = tuple(sorted({c.target_index() for c in pool}))
            candidates = tuple(sorted({text_signature(c.option()) for c in pool if c.option()}))

            result.append(
                CategoryBucket(
                    category=category,
                    sources=sources,
                    candidate_types=candidate_types,
                    instructions=instructions,
                    candidate_counts=candidate_counts,
                    target_indices=target_indices,
                    candidates=candidates,
                )
            )

        logger.debug(f"Created {len(result)} category buckets across {len(cases)} cases.")
        return tuple(result)

__all__ = ["text_signature", "CategoryBucket", "IBucketPolicy", "CaseBucketPolicy"]
