import random
from typing import Optional, Protocol, Sequence, runtime_checkable
from data.case import DataSplits, DecisionCase
from data.split.stratification import IStratificationPolicy, RoundRobinStratification

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
            pre_test = [c for c in items if c.split == "test"]
            pre_val = [c for c in items if c.split == "val"]
            pool = [c for c in items if c.split not in ("test", "val")]

            if pre_test:
                test_all.extend(self.stratifier.balance(pre_test, self.max_test_per_category, rng))
            elif pool and self.max_test_per_category > 0:
                selected_test = self.stratifier.balance(pool, min(len(pool), self.max_test_per_category), rng)
                test_all.extend(selected_test)
                pool = [c for c in pool if id(c) not in {id(t) for t in selected_test}]

            if pre_val:
                val_all.extend(self.stratifier.balance(pre_val, self.max_val_per_category, rng))
            elif pool and self.max_val_per_category > 0:
                selected_val = self.stratifier.balance(pool, min(len(pool), self.max_val_per_category), rng)
                val_all.extend(selected_val)
                pool = [c for c in pool if id(c) not in {id(v) for v in selected_val}]

            if pool and self.max_train_per_category > 0:
                train_all.extend(self.stratifier.balance(pool, min(len(pool), self.max_train_per_category), rng))

        return DataSplits(train=train_all, val=val_all, test=test_all)

__all__ = ["ISplitPolicy", "BudgetSplitPolicy"]
