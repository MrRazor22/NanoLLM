from dataclasses import replace
import logging
import random
from typing import Optional, Protocol, Sequence, runtime_checkable
from data.case import DataSplits, DecisionCase
from data.split.stratification import IStratificationPolicy, RoundRobinStratification

logger = logging.getLogger(__name__)

@runtime_checkable
class ISplitPolicy(Protocol):
    """Sovereign policy contract: partitions source records into train, val, and test splits."""
    def partition(self, cases: Sequence[DecisionCase]) -> DataSplits: ...

class BudgetSplitPolicy(ISplitPolicy):
    """Partitions source cases with symmetric per-category caps and balanced class stratification."""

    def __init__(
        self,
        max_train_per_category: int = 1500,
        max_val_per_category: int = 150,
        max_test_per_category: int = 400,
        stratifier: Optional[IStratificationPolicy] = None,
        seed: int = 42,
    ):
        self.max_train_per_category = max_train_per_category
        self.max_val_per_category = max_val_per_category
        self.max_test_per_category = max_test_per_category
        self.stratifier = stratifier or RoundRobinStratification()
        self.seed = seed

    def partition(self, cases: Sequence[DecisionCase]) -> DataSplits:
        rng = random.Random(self.seed)

        cases_by_cat: dict[str, list[DecisionCase]] = {}
        for c in cases:
            if not c.category:
                raise ValueError(f"DecisionCase missing mandatory category tag from source '{c.source}'.")
            cases_by_cat.setdefault(c.category, []).append(c)

        train_all: list[DecisionCase] = []
        val_all: list[DecisionCase] = []
        test_all: list[DecisionCase] = []

        for cat, items in cases_by_cat.items():
            tests = [c for c in items if c.split == "test"]
            trains = [c for c in items if c.split != "test"]

            if tests:
                test_all.extend(replace(c, split="test") for c in self.stratifier.balance(tests, self.max_test_per_category, rng))

            if trains:
                quota = self.max_val_per_category + self.max_train_per_category
                selected = self.stratifier.balance(trains, min(len(trains), quota), rng)
                val_slice = selected[:self.max_val_per_category]
                train_slice = selected[self.max_val_per_category:]

                val_all.extend(replace(c, split="val") for c in val_slice)
                train_all.extend(replace(c, split="train") for c in train_slice)

        splits = DataSplits(train=train_all, val=val_all, test=test_all)
        logger.info("Budget partition: %d train, %d val, %d test across %d categories", len(train_all), len(val_all), len(test_all), len(cases_by_cat))
        return splits

__all__ = ["ISplitPolicy", "BudgetSplitPolicy"]
