from dataclasses import replace
import logging
import random
from typing import Protocol, Sequence, runtime_checkable

from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class ISplitPolicy(Protocol):
    def partition(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]: ...

class RatioSplitPolicy(ISplitPolicy):
    def __init__(self, train_ratio: float = 0.80, val_ratio: float = 0.20, seed: int = 42):
        if train_ratio <= 0.0 or val_ratio <= 0.0:
            raise ValueError("Split ratios must be positive.")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

    def partition(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]:
        if not cases:
            return ()

        rng = random.Random(self.seed)
        by_cat: dict[str, list[DecisionCase]] = {}
        for c in cases:
            cat = c.meta.category if c.meta else ""
            by_cat.setdefault(cat, []).append(c)

        partitioned: list[DecisionCase] = []
        for cat, items in by_cat.items():
            author_tests = [c for c in items if c.meta and c.meta.split == "test"]
            pool = [c for c in items if not (c.meta and c.meta.split == "test")]

            rng.shuffle(pool)
            total = self.train_ratio + self.val_ratio
            val_count = max(1, int(len(pool) * (self.val_ratio / total))) if len(pool) > 1 else 0

            val_cases = pool[:val_count]
            train_cases = pool[val_count:]

            partitioned.extend(author_tests)
            partitioned.extend(replace(c, meta=replace(c.meta, split="val")) for c in val_cases)
            partitioned.extend(replace(c, meta=replace(c.meta, split="train")) for c in train_cases)

        logger.info("Partitioned %d cases across %d categories into train and val", len(partitioned), len(by_cat))
        return partitioned

__all__ = ["ISplitPolicy", "RatioSplitPolicy"]
