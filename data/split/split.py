from dataclasses import replace
import logging
import random
from typing import Optional, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase
from data.split.policies.balancer import ICaseBalancer, RoundRobinCaseBalancer
from data.split.policies.leakage import ILeakagePolicy, JaccardLeakagePolicy

logger = logging.getLogger(__name__)

@runtime_checkable
class ISplitPolicy(Protocol):
    def partition(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]: ...

class BudgetSplitPolicy(ISplitPolicy):
    def __init__(
        self,
        max_train: int = 1500,
        max_val: int = 150,
        max_test: int = 400,
        balancer: Optional[ICaseBalancer] = None,
        leakage: Optional[ILeakagePolicy] = None,
        seed: int = 42,
    ):
        self.max_train = max_train
        self.max_val = max_val
        self.max_test = max_test
        self.balancer = balancer or RoundRobinCaseBalancer()
        self.leakage = leakage or JaccardLeakagePolicy()
        self.seed = seed

    def partition(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]:
        rng = random.Random(self.seed)
        by_cat: dict[str, list[DecisionCase]] = {}
        for c in cases:
            by_cat.setdefault(c.meta.category, []).append(c)

        train_all: list[DecisionCase] = []
        val_all: list[DecisionCase] = []
        test_all: list[DecisionCase] = []

        for cat, items in by_cat.items():
            tests = [c for c in items if c.meta.split == "test"]
            cat_test = list(self.balancer.balance(tests, self.max_test, rng)) if tests else []

            pool = [c for c in items if c.meta.split != "test"]
            balanced_pool = self.balancer.balance(pool, len(pool), rng)
            clean_pool = self.leakage.decontaminate(balanced_pool, cat_test) if cat_test else balanced_pool

            target_quota = self.max_val + self.max_train
            selected = clean_pool[:target_quota]

            test_all.extend(replace(c, meta=replace(c.meta, split="test")) for c in cat_test)
            val_all.extend(replace(c, meta=replace(c.meta, split="val")) for c in selected[:self.max_val])
            train_all.extend(replace(c, meta=replace(c.meta, split="train")) for c in selected[self.max_val:])

        logger.info("Budget partition: %d train, %d val, %d test across %d categories", len(train_all), len(val_all), len(test_all), len(by_cat))
        return [*train_all, *val_all, *test_all]

__all__ = ["ISplitPolicy", "BudgetSplitPolicy"]
