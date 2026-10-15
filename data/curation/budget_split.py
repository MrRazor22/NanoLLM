import random
from typing import Optional, Protocol, Sequence, runtime_checkable
from data.case import DataSplits, DecisionCase, SourceRecord

@runtime_checkable
class ISplitPolicy(Protocol):
    """Sovereign policy contract: partitions source records into train, val, and test splits."""
    def partition(self, records: Sequence[SourceRecord]) -> DataSplits: ...

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

    def partition(self, records: Sequence[SourceRecord]) -> DataSplits:
        rng = random.Random(self.seed)
        train_recs: list[DecisionCase] = []
        val_recs: list[DecisionCase] = []
        test_recs: list[DecisionCase] = []

        for r in records:
            if r.tag == "test":
                test_recs.append(r.sample)
            elif r.tag == "val":
                val_recs.append(r.sample)
            elif r.tag == "train":
                train_recs.append(r.sample)
            else:
                raise ValueError(f"Invalid split tag '{r.tag}'. Must be 'train', 'val', or 'test'.")

        if self.test_quota_per_category is not None and test_recs:
            by_category: dict[str, list[DecisionCase]] = {}
            for c in test_recs:
                by_category.setdefault(c.category or "general", []).append(c)

            balanced_test_recs: list[DecisionCase] = []
            for cat, cat_items in by_category.items():
                if len(cat_items) <= self.test_quota_per_category:
                    balanced_test_recs.extend(cat_items)
                    continue

                if not self.balanced_test:
                    rng.shuffle(cat_items)
                    balanced_test_recs.extend(cat_items[: self.test_quota_per_category])
                    continue

                by_class: dict[str, list[DecisionCase]] = {}
                for item in cat_items:
                    target_val = str(next(iter(item.gold.values()))) if item.gold else "default"
                    by_class.setdefault(target_val, []).append(item)

                for group in by_class.values():
                    rng.shuffle(group)

                target_per_class = max(1, self.test_quota_per_category // max(1, len(by_class)))
                if len(by_class) == 2:
                    min_available = min(len(g) for g in by_class.values())
                    per_class = min(target_per_class, max(1, min_available))
                    selected: list[DecisionCase] = []
                    for group in by_class.values():
                        selected.extend(group[:per_class])
                else:
                    selected = []
                    for group in by_class.values():
                        selected.extend(group[:target_per_class])
                    if len(selected) < self.test_quota_per_category:
                        remaining = [item for group in by_class.values() for item in group[target_per_class:]]
                        rng.shuffle(remaining)
                        selected.extend(remaining[: self.test_quota_per_category - len(selected)])

                balanced_test_recs.extend(selected)

            test_recs = balanced_test_recs

        if self.train_quota_per_category is not None and train_recs:
            by_train_cat: dict[str, list[DecisionCase]] = {}
            for c in train_recs:
                by_train_cat.setdefault(c.category or "general", []).append(c)

            balanced_train: list[DecisionCase] = []
            for cat_items in by_train_cat.values():
                if len(cat_items) <= self.train_quota_per_category:
                    balanced_train.extend(cat_items)
                else:
                    rng.shuffle(cat_items)
                    balanced_train.extend(cat_items[: self.train_quota_per_category])
            train_recs = balanced_train

        needed_val = max(0, self.val_quota - len(val_recs))
        if needed_val > 0 and train_recs:
            take = min(needed_val, max(1, len(train_recs) // 5)) if len(train_recs) <= needed_val else needed_val
            rng.shuffle(train_recs)
            val_recs.extend(train_recs[:take])
            train_recs = train_recs[take:]

        if len(train_recs) > self.train_quota:
            rng.shuffle(train_recs)
            train_recs = train_recs[:self.train_quota]

        if self.test_quota and len(test_recs) > self.test_quota:
            rng.shuffle(test_recs)
            test_recs = test_recs[:self.test_quota]

        return DataSplits(train=train_recs, val=val_recs, test=test_recs)
