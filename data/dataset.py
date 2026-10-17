from typing import List, Optional, Protocol, Sequence, runtime_checkable

from data.case import DataSplits, DecisionCase, SourceRecord, SplitTag
from data.curation.budget_split import BudgetSplitPolicy, ISplitPolicy
from data.curation.ngram_leakage import ILeakagePolicy, NGramLeakagePolicy
from data.registry import AVAILABLE_SOURCES
from data.sources.source import IDataSource
from data.storage_policy import DiskStoragePolicy, IStoragePolicy

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract: manages dataset source preparation, leakage audit, and split access."""
    def prepare(self) -> DataSplits: ...
    def load(self, split: SplitTag = "train") -> Sequence[DecisionCase]: ...

class Dataset(IDataset):
    """Sovereign Core Primitive: manages source registration, split preparation, zero leakage, and in-memory access."""

    def __init__(
        self,
        split_policy: Optional[ISplitPolicy] = None,
        leakage_policy: Optional[ILeakagePolicy] = None,
        storage_policy: Optional[IStoragePolicy] = None,
        sources: Optional[Sequence[IDataSource]] = None,
    ):
        self.split_policy = split_policy or BudgetSplitPolicy()
        self.leakage_policy = leakage_policy or NGramLeakagePolicy()
        self.storage = storage_policy or DiskStoragePolicy()
        self.sources: List[IDataSource] = list(sources) if sources is not None else [src() for src in AVAILABLE_SOURCES.values()]
        self.leak_audit: dict[str, int] = {}
        self._splits: Optional[DataSplits] = None

    def prepare(self) -> DataSplits:
        """Extracts all registered sources, partitions splits, audits for leakage, and holds splits in memory."""
        if not self.sources:
            raise ValueError("No data sources registered for dataset preparation.")

        records: List[SourceRecord] = []
        for src in self.sources:
            records.extend(src.read())

        splits = self.split_policy.partition(records)

        # Enforce zero-leakage invariant: prune candidate training and validation samples overlapping holdouts
        test_leaks = list(self.leakage_policy.find_leaks(splits.train, splits.test))
        val_leaks = list(self.leakage_policy.find_leaks(splits.train, splits.val))
        val_test_leaks = list(self.leakage_policy.find_leaks(splits.val, splits.test))

        unique_val_test = {id(c): c for c in val_test_leaks}
        unique_train_leaks = {id(c): c for c in (test_leaks + val_leaks)}
        if unique_val_test or unique_train_leaks:
            splits = DataSplits(
                train=[c for c in splits.train if id(c) not in unique_train_leaks],
                val=[c for c in splits.val if id(c) not in unique_val_test],
                test=splits.test,
            )

        from collections import Counter
        all_unique_leaks = {**unique_train_leaks, **unique_val_test}
        self.leak_audit = {
            "total_unique": len(all_unique_leaks),
            "test_leaks": len(test_leaks),
            "val_leaks": len(val_leaks),
            "val_test_leaks": len(val_test_leaks),
            "by_category": dict(Counter(c.category or "unknown" for c in all_unique_leaks.values())),
        }

        if self.leakage_policy.find_leaks(splits.train, splits.test):
            raise ValueError("Contamination invariant violated: train split leaks test holdout.")
        if self.leakage_policy.find_leaks(splits.train, splits.val):
            raise ValueError("Contamination invariant violated: train split leaks validation.")
        if self.leakage_policy.find_leaks(splits.val, splits.test):
            raise ValueError("Contamination invariant violated: val split leaks test holdout.")

        self._splits = splits
        self.storage.save(splits)
        return splits

    def load(self, split: SplitTag = "train") -> Sequence[DecisionCase]:
        """Loads domain decision cases from in-memory splits or persistent storage policy."""
        if self._splits is not None:
            return getattr(self._splits, split)
        return self.storage.load(split)

__all__ = ["IDataset", "Dataset"]
