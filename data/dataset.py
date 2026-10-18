from typing import List, Optional, Protocol, Sequence, runtime_checkable

from data.case import DataSplits, DecisionCase
from data.policies.leakage import ILeakagePolicy, NGramLeakagePolicy
from data.policies.storage import DiskStoragePolicy, IStoragePolicy
from data.split.split import BudgetSplitPolicy, ISplitPolicy
from data.sources.source import IDataSource, get_available_sources

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract: delivers partitioned dataset splits."""
    def build(self) -> DataSplits: ...
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
        self.sources: Sequence[IDataSource] = sources if sources is not None else get_available_sources()
        self.leak_audit: dict[str, int] = {}
        self._splits: Optional[DataSplits] = None

    def build(self) -> DataSplits:
        """Forces fresh ingestion from all sources, partitions splits, enforces zero-leakage, and persists corpus."""
        if not self.sources:
            raise ValueError("No data sources registered for dataset preparation.")

        raw_cases: List[DecisionCase] = []
        for src in self.sources:
            raw_cases.extend(src.cases())

        splits = self.split_policy.partition(raw_cases)

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
        """Delivers cached or stored splits, automatically building from sources if not yet persisted."""
        if self._splits is not None:
            return self._splits
        try:
            self._splits = self.storage.load()
            return self._splits
        except FileNotFoundError:
            return self.build()

__all__ = ["IDataset", "Dataset"]
