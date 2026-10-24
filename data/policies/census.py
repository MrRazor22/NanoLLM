from collections import defaultdict
from dataclasses import dataclass, field
import logging
from typing import Mapping, Protocol, Self, Sequence, runtime_checkable

from data.case import DecisionCase, text_signature

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class CategoryBucket:
    category: str
    sources: tuple[str, ...]
    candidate_types: tuple[str, ...]
    instructions: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    target_indices: tuple[int, ...]
    candidates: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class DatasetCensus:
    categories: Mapping[str, CategoryBucket] = field(default_factory=dict)
    holdouts: tuple[frozenset[int], ...] = ()

    def union(self, other: Self) -> Self:
        merged_categories = dict(self.categories)
        for cat, b in other.categories.items():
            if cat in merged_categories:
                cur = merged_categories[cat]
                merged_categories[cat] = CategoryBucket(
                    category=cat,
                    sources=tuple(sorted(set(cur.sources) | set(b.sources))),
                    candidate_types=tuple(sorted(set(cur.candidate_types) | set(b.candidate_types))),
                    instructions=tuple(sorted(set(cur.instructions) | set(b.instructions))),
                    candidate_counts=tuple(sorted(set(cur.candidate_counts) | set(b.candidate_counts))),
                    target_indices=tuple(sorted(set(cur.target_indices) | set(b.target_indices))),
                    candidates=tuple(sorted(set(cur.candidates) | set(b.candidates))),
                )
            else:
                merged_categories[cat] = b

        merged_holdouts = tuple(sorted(set(self.holdouts) | set(other.holdouts), key=lambda s: (len(s), sorted(s))))
        return DatasetCensus(categories=merged_categories, holdouts=merged_holdouts)

@runtime_checkable
class ICensusPolicy(Protocol):
    def create(self, cases: Sequence[DecisionCase]) -> DatasetCensus: ...

class CaseCensusPolicy(ICensusPolicy):
    def create(self, cases: Sequence[DecisionCase]) -> DatasetCensus:
        if not cases:
            return DatasetCensus()

        by_category: dict[str, list[DecisionCase]] = defaultdict(list)
        holdout_set: set[frozenset[int]] = set()

        for c in cases:
            if c.meta.split == "test":
                if h := c.prompt_hashes():
                    holdout_set.add(frozenset(h))
            else:
                by_category[c.meta.category].append(c)

        categories: dict[str, CategoryBucket] = {}
        for cat, pool in sorted(by_category.items(), key=lambda item: item[0]):
            categories[cat] = CategoryBucket(
                category=cat,
                sources=tuple(sorted({c.meta.source for c in pool})),
                candidate_types=tuple(sorted({c.meta.candidate_type for c in pool})),
                instructions=tuple(sorted({sig for c in pool if (sig := c.instruction_signature())})),
                candidate_counts=tuple(sorted({len(c.candidates) for c in pool})),
                target_indices=tuple(sorted({c.target_index() for c in pool})),
                candidates=tuple(sorted({sig for c in pool if (sig := c.option_signature())})),
            )

        return DatasetCensus(
            categories=categories,
            holdouts=tuple(sorted(holdout_set, key=lambda s: (len(s), sorted(s)))),
        )

__all__ = ["text_signature", "CategoryBucket", "DatasetCensus", "ICensusPolicy", "CaseCensusPolicy"]
