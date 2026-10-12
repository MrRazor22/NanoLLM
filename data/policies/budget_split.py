import random
from typing import Optional, Protocol, Sequence, runtime_checkable
from data.case import DataSplits, DecisionCase, SourceRecord

@runtime_checkable
class ISplitPolicy(Protocol):
    """Sovereign policy contract: partitions source records into train, val, and test splits."""
    @property
    def signature(self) -> str: ...
    def partition(self, records: Sequence[SourceRecord]) -> DataSplits: ...

class BudgetSplitPolicy(ISplitPolicy):
    """Partitions source records by preserving explicit holdout tags and splitting pools."""
    def __init__(
        self,
        train_quota: int = 40000,
        val_quota: int = 2500,
        test_quota: Optional[int] = None,
        seed: int = 42,
    ):
        self.train_quota = train_quota
        self.val_quota = val_quota
        self.test_quota = test_quota
        self.seed = seed

    @property
    def signature(self) -> str:
        import hashlib
        import json
        payload = {
            "type": "BudgetSplitPolicy",
            "train_quota": self.train_quota,
            "val_quota": self.val_quota,
            "test_quota": self.test_quota,
            "seed": self.seed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]

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

        # Carve validation holdout from train if needed without starving training pool
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

__all__ = ["ISplitPolicy", "BudgetSplitPolicy"]
