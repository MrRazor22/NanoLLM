import random
from typing import Optional, Protocol, Sequence, runtime_checkable
from data.case import DataSplits, DecisionCase

@runtime_checkable
class ISplitPolicy(Protocol):
    """Sovereign policy contract: partitions source records into train, val, and test splits."""
    def partition(self, cases: Sequence[DecisionCase]) -> DataSplits: ...

def _group_by(items: Sequence[DecisionCase], key_fn) -> list[list[DecisionCase]]:
    """Pure domain transform: clusters cases into groups by key without leaking raw dictionary keys."""
    groups: dict[str, list[DecisionCase]] = {}
    for item in items:
        groups.setdefault(key_fn(item), []).append(item)
    return list(groups.values())

def _case_label(case: DecisionCase) -> str:
    """Pure domain transform: extracts leading target label for stratification."""
    ans = case.option.answer
    if ans is None: return "default"
    tgt = getattr(ans, "choice", getattr(ans, "decision", getattr(ans, "score", None)))
    return str(tgt) if tgt is not None else "default"

def _balance_classes(groups: list[list[DecisionCase]], quota: int, rng: random.Random) -> list[DecisionCase]:
    """Pure domain transform: balances items across label class groups up to specified quota."""
    for g in groups:
        rng.shuffle(g)
    target_per_class = max(1, quota // max(1, len(groups)))
    per_class = min(target_per_class, min(len(g) for g in groups)) if len(groups) == 2 else target_per_class
    selected = [item for g in groups for item in g[:per_class]]
    if len(groups) != 2 and len(selected) < quota:
        remaining = [item for g in groups for item in g[target_per_class:]]
        rng.shuffle(remaining)
        selected.extend(remaining[: quota - len(selected)])
    return selected

class BudgetSplitPolicy(ISplitPolicy):
    """Partitions source records by preserving explicit holdout tags, stratifying holdouts, and balancing pools."""
    def __init__(
        self,
        train_quota: int = 40000,
        train_quota_per_category: Optional[int] = 1500,
        val_quota: int = 2500,
        test_quota: Optional[int] = None,
        test_quota_per_category: Optional[int] = 400,
        balanced_test: bool = True,
        seed: int = 42,
    ):
        self.train_quota = train_quota
        self.train_quota_per_category = train_quota_per_category
        self.val_quota = val_quota
        self.test_quota = test_quota
        self.test_quota_per_category = test_quota_per_category
        self.balanced_test = balanced_test
        self.seed = seed

    def partition(self, cases: Sequence[DecisionCase]) -> DataSplits:
        rng = random.Random(self.seed)
        train: list[DecisionCase] = [c for c in cases if c.split == "train" or c.split is None]
        val: list[DecisionCase] = [c for c in cases if c.split == "val"]
        test: list[DecisionCase] = [c for c in cases if c.split == "test"]

        if self.test_quota_per_category is not None and test:
            balanced_test: list[DecisionCase] = []
            for items in _group_by(test, lambda c: c.category or "general"):
                if len(items) <= self.test_quota_per_category:
                    balanced_test.extend(items)
                elif not self.balanced_test:
                    rng.shuffle(items)
                    balanced_test.extend(items[: self.test_quota_per_category])
                else:
                    balanced_test.extend(_balance_classes(_group_by(items, _case_label), self.test_quota_per_category, rng))
            test = balanced_test

        if self.train_quota_per_category is not None and train:
            balanced_train: list[DecisionCase] = []
            for items in _group_by(train, lambda c: c.category or "general"):
                rng.shuffle(items)
                balanced_train.extend(items[: self.train_quota_per_category])
            train = balanced_train

        needed_val = max(0, self.val_quota - len(val))
        if needed_val > 0 and train:
            take = min(needed_val, max(1, len(train) // 5)) if len(train) <= needed_val else needed_val
            rng.shuffle(train)
            val.extend(train[:take])
            train = train[take:]

        if len(train) > self.train_quota:
            rng.shuffle(train)
            train = train[: self.train_quota]

        if self.test_quota and len(test) > self.test_quota:
            rng.shuffle(test)
            test = test[: self.test_quota]

        return DataSplits(train=train, val=val, test=test)
