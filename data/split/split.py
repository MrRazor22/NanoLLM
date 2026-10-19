from dataclasses import replace
import itertools
import logging
import random
from typing import Optional, Protocol, Sequence, runtime_checkable
from data.case import DataSplits, DecisionCase
from data.split.policies.leakage import ILeakagePolicy, JaccardLeakagePolicy
from data.split.policies.stratification import IStratificationPolicy, RoundRobinStratification

logger = logging.getLogger(__name__)

@runtime_checkable
class ISplitPolicy(Protocol):
    """Sovereign policy contract: partitions source records into train, val, and test splits."""
    def partition(self, cases: Sequence[DecisionCase]) -> DataSplits: ...

class BudgetSplitPolicy(ISplitPolicy):
    """Partitions cases per-category with balanced stratification and lazy leakage decontamination."""

    def __init__(
        self,
        max_train: int = 1500,
        max_val: int = 150,
        max_test: int = 400,
        stratifier: Optional[IStratificationPolicy] = None,
        leakage: Optional[ILeakagePolicy] = None,
        seed: int = 42,
    ):
        self.max_train = max_train
        self.max_val = max_val
        self.max_test = max_test
        self.stratifier = stratifier or RoundRobinStratification()
        self.leakage = leakage or JaccardLeakagePolicy()
        self.seed = seed

    def partition(self, cases: Sequence[DecisionCase]) -> DataSplits:
        rng = random.Random(self.seed)
        by_cat: dict[str, list[DecisionCase]] = {}
        for c in cases:
            by_cat.setdefault(c.category, []).append(c)

        train_all: list[DecisionCase] = []
        val_all: list[DecisionCase] = []
        test_all: list[DecisionCase] = []

        for cat, items in by_cat.items():
            tests = [c for c in items if c.split == "test"]
            cat_test = self.stratifier.balance(tests, self.max_test, rng) if tests else []

            # Stream balanced candidates, filter leaks lazily, and take target quota
            pool = [c for c in items if c.split != "test"]
            balanced_stream = self.stratifier.stream(pool, rng)
            clean_stream = self.leakage.filter(balanced_stream, cat_test)

            target_quota = self.max_val + self.max_train
            selected = list(itertools.islice(clean_stream, target_quota))

            test_all.extend(replace(c, split="test") for c in cat_test)
            val_all.extend(replace(c, split="val") for c in selected[:self.max_val])
            train_all.extend(replace(c, split="train") for c in selected[self.max_val:])

        logger.info("Budget partition: %d train, %d val, %d test across %d categories", len(train_all), len(val_all), len(test_all), len(by_cat))
        return DataSplits(train=train_all, val=val_all, test=test_all)

__all__ = ["ISplitPolicy", "BudgetSplitPolicy"]
