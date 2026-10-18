from typing import List, Optional, Protocol, Sequence, runtime_checkable

from data.case import DataSplits, DecisionCase
from data.curation.budget_split import BudgetSplitPolicy, ISplitPolicy
from data.curation.ngram_leakage import ILeakagePolicy, NGramLeakagePolicy
from data.sources.source import IDataSource, get_available_sources
from data.storage_policy import DiskStoragePolicy, IStoragePolicy

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract: manages dataset source preparation, leakage audit, and split access."""
    def prepare(self) -> DataSplits: ...
    def load(self) -> DataSplits: ...

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
        self.sources: List[IDataSource] = list(sources) if sources is not None else [src() for src in get_available_sources().values()]
        self.leak_audit: dict[str, int] = {}
        self._splits: Optional[DataSplits] = None

    def prepare(self) -> DataSplits:
        if not self.sources:
            raise ValueError("No data sources registered for dataset preparation.")

        raw_cases: List[DecisionCase] = []
        for src in self.sources:
            raw_cases.extend(src.read())

        splits = self.split_policy.partition(raw_cases)

        # Enforce zero-leakage invariant by filtering candidates against holdouts
        filtered_val = self.leakage_policy.filter(splits.val, splits.test)
        filtered_train = self.leakage_policy.filter(splits.train, splits.test)
        filtered_train = self.leakage_policy.filter(filtered_train, filtered_val)

        pruned_train = len(splits.train) - len(filtered_train)
        pruned_val = len(splits.val) - len(filtered_val)

        splits = DataSplits(train=filtered_train, val=filtered_val, test=splits.test)

        self.leak_audit = {
            "total_unique": pruned_train + pruned_val,
            "pruned_train": pruned_train,
            "pruned_val": pruned_val,
        }

        self._splits = splits
        self.storage.save(splits)
        return splits

    def load(self) -> DataSplits:
        """Loads partitioned DataSplits from memory cache or persistent storage."""
        if self._splits is not None:
            return self._splits
        self._splits = self.storage.load()
        return self._splits

__all__ = ["IDataset", "Dataset"]
