from dataclasses import dataclass
import random
from typing import Protocol, Sequence, runtime_checkable
from inference.case import DecisionCase
from data.sources.source import SourceRecord

@dataclass(frozen=True)
class DataSplits:
    """Consolidated partition of domain decision cases."""
    train: Sequence[DecisionCase]
    val: Sequence[DecisionCase]
    test: Sequence[DecisionCase]

@runtime_checkable
class ISplitPolicy(Protocol):
    """Sovereign policy contract: partitions source records into train, val, and test splits."""
    def partition(self, records: Sequence[SourceRecord]) -> DataSplits: ...

class BudgetSplitPolicy(ISplitPolicy):
    """Partitions source records by preserving explicit holdout tags and splitting pools."""
    def __init__(
        self,
        train_quota: int = 40000,
        val_quota: int = 2500,
        test_quota: int = 2500,
        seed: int = 42,
    ):
        self.train_quota = train_quota
        self.val_quota = val_quota
        self.test_quota = test_quota
        self.seed = seed

    def partition(self, records: Sequence[SourceRecord]) -> DataSplits:
        rng = random.Random(self.seed)
        train_recs: list[DecisionCase] = []
        val_recs: list[DecisionCase] = []
        test_recs: list[DecisionCase] = []
        unsplit: list[DecisionCase] = []

        for r in records:
            if r.tag == "test":
                test_recs.append(r.sample)
            elif r.tag == "val":
                val_recs.append(r.sample)
            elif r.tag == "train":
                train_recs.append(r.sample)
            else:
                unsplit.append(r.sample)

        if unsplit:
            rng.shuffle(unsplit)
            needed_test = max(0, self.test_quota - len(test_recs))
            test_recs.extend(unsplit[:needed_test])
            remainder = unsplit[needed_test:]

            needed_val = max(0, self.val_quota - len(val_recs))
            val_recs.extend(remainder[:needed_val])
            train_recs.extend(remainder[needed_val:])

        if len(train_recs) > self.train_quota:
            rng.shuffle(train_recs)
            train_recs = train_recs[:self.train_quota]

        return DataSplits(train=train_recs, val=val_recs, test=test_recs)

__all__ = ["DataSplits", "ISplitPolicy", "BudgetSplitPolicy"]
