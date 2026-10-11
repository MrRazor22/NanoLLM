from pathlib import Path
from typing import List, Optional, Protocol, Sequence, runtime_checkable

from inference.case import DecisionCase
from data.policies.cache import DiskCachePolicy, ICachePolicy
from data.policies.leakage import ILeakagePolicy, NGramLeakagePolicy
from data.policies.split import BudgetSplitPolicy, DataSplits, ISplitPolicy
from data.sources import IDataSource, SourceRecord, standard_sources

DATA_DIR = Path(__file__).resolve().parent
CORPUS_DIR = DATA_DIR / "corpus"

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract: manages dataset source preparation, leakage audit, and cached loading."""
    def register(self, source: IDataSource) -> "IDataset": ...
    def prepare(self, sources: Optional[Sequence[IDataSource]] = None) -> DataSplits: ...
    def load(self, split: str = "train") -> Sequence[DecisionCase]: ...

class Dataset(IDataset):
    """Sovereign Core Primitive: manages source registration, split preparation, zero leakage, and cached loading."""

    def __init__(
        self,
        split_policy: Optional[ISplitPolicy] = None,
        leakage_policy: Optional[ILeakagePolicy] = None,
        cache_policy: Optional[ICachePolicy] = None,
        sources: Optional[Sequence[IDataSource]] = None,
    ):
        self.split_policy = split_policy or BudgetSplitPolicy()
        self.leakage_policy = leakage_policy or NGramLeakagePolicy()
        self.cache_policy = cache_policy or DiskCachePolicy()
        self._sources: List[IDataSource] = list(sources) if sources is not None else standard_sources()

    def register(self, source: IDataSource) -> "Dataset":
        """Registers a data source into the dataset collection."""
        self._sources.append(source)
        return self

    def prepare(self, sources: Optional[Sequence[IDataSource]] = None) -> DataSplits:
        """Extracts all registered sources, partitions splits, audits for leakage, and caches via cache policy."""
        src_list = list(sources) if sources is not None else self._sources
        if not src_list:
            raise ValueError("No data sources registered for dataset preparation.")

        records: List[SourceRecord] = []
        for src in src_list:
            records.extend(src.read())

        splits = self.split_policy.partition(records)

        # Enforce zero-leakage invariant: prune candidate training samples overlapping holdouts
        leaks = set(id(c) for c in self.leakage_policy.find_leaks(splits.train, splits.test))
        leaks.update(id(c) for c in self.leakage_policy.find_leaks(splits.train, splits.val))
        if leaks:
            splits = DataSplits(
                train=[c for c in splits.train if id(c) not in leaks],
                val=splits.val,
                test=splits.test,
            )

        if self.leakage_policy.find_leaks(splits.train, splits.test):
            raise ValueError("Contamination invariant violated: train split leaks test holdout.")
        if self.leakage_policy.find_leaks(splits.train, splits.val):
            raise ValueError("Contamination invariant violated: train split leaks validation.")

        # Delegate caching to injected cache policy
        self.cache_policy.put("train", splits.train)
        self.cache_policy.put("val", splits.val)
        self.cache_policy.put("test", splits.test)

        return splits

    def load(self, split: str = "train") -> Sequence[DecisionCase]:
        """Loads domain decision cases from cache; self-prepares from sources on cache miss."""
        cached = self.cache_policy.get(split)
        if cached is not None:
            return cached
        if self._sources:
            self.prepare()
            cached = self.cache_policy.get(split)
            if cached is not None:
                return cached
        raise FileNotFoundError(
            f"Corpus partition '{split}' not found and no data sources available to prepare it."
        )

__all__ = ["DATA_DIR", "CORPUS_DIR", "IDataset", "Dataset", "DataSplits"]
