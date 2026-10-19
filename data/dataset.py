import logging
from typing import List, Optional, Protocol, Sequence, runtime_checkable

from data.case import DataSplits, DecisionCase
from data.storage import DiskStoragePolicy, IStoragePolicy
from data.split.split import BudgetSplitPolicy, ISplitPolicy
from data.sources.source import IDataSource, get_available_sources

logger = logging.getLogger(__name__)

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract: delivers partitioned dataset splits."""
    def build(self) -> DataSplits: ...
    def load(self) -> DataSplits: ...

class Dataset(IDataset):
    """Sovereign Core Primitive: manages source registration, split preparation, and in-memory access."""

    def __init__(
        self,
        split_policy: Optional[ISplitPolicy] = None,
        storage_policy: Optional[IStoragePolicy] = None,
        sources: Optional[Sequence[IDataSource]] = None,
    ):
        self.split_policy = split_policy or BudgetSplitPolicy()
        self.storage = storage_policy or DiskStoragePolicy()
        self.sources: Sequence[IDataSource] = sources if sources is not None else get_available_sources()
        self._splits: Optional[DataSplits] = None

    def build(self) -> DataSplits:
        if not self.sources:
            raise ValueError("No data sources registered for dataset preparation.")

        raw_cases: List[DecisionCase] = []
        for src in self.sources:
            raw_cases.extend(src.cases())
        logger.info("Ingested %d raw cases across %d sources", len(raw_cases), len(self.sources))
        splits = self.split_policy.partition(raw_cases)
        self._splits = splits
        self.storage.save(splits)
        return splits

    def load(self) -> DataSplits:
        if self._splits is not None:
            return self._splits
        try:
            self._splits = self.storage.load()
            return self._splits
        except FileNotFoundError:
            return self.build()

__all__ = ["IDataset", "Dataset"]
