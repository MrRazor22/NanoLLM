from pathlib import Path
from typing import List, Optional, Protocol, Sequence, runtime_checkable

from inference.questions import DecisionSample
from data.policies.cache import DiskCachePolicy, ICachePolicy
from data.policies.leakage import ILeakagePolicy, NGramLeakagePolicy
from data.policies.split import BudgetSplitPolicy, DataSplits, ISplitPolicy
from data.sources.source import IDataSource, SourceRecord

DATA_DIR = Path(__file__).resolve().parent
CORPUS_DIR = DATA_DIR / "corpus"

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract: supplies partitioned decision samples to callers."""
    def load(self, split: str = "train") -> Sequence[DecisionSample]: ...

class Dataset(IDataset):
    """Sovereign Core Primitive: manages source registration, split preparation, zero leakage, and cached loading."""

    def __init__(
        self,
        split_policy: Optional[ISplitPolicy] = None,
        leakage_policy: Optional[ILeakagePolicy] = None,
        cache_policy: Optional[ICachePolicy] = None,
        seed: int = 42,
    ):
        self.split_policy = split_policy or BudgetSplitPolicy(seed=seed)
        self.leakage_policy = leakage_policy or NGramLeakagePolicy(13)
        self.cache_policy = cache_policy or DiskCachePolicy(root_dir=CORPUS_DIR)
        self.seed = seed
        self._sources: List[IDataSource] = []

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

        # Invariant verification: zero leakage
        if self.leakage_policy.find_leaks(splits.train, splits.test):
            raise ValueError("Contamination invariant violated: train split leaks test holdout.")
        if self.leakage_policy.find_leaks(splits.train, splits.val):
            raise ValueError("Contamination invariant violated: train split leaks validation.")

        # Delegate caching to injected cache policy
        self.cache_policy.put("train", splits.train)
        self.cache_policy.put("val", splits.val)
        self.cache_policy.put("test", splits.test)

        return splits

    def load(self, split: str = "train") -> Sequence[DecisionSample]:
        """Loads samples from cache policy; triggers preparation automatically on cache miss."""
        cached = self.cache_policy.get(split)
        if cached is not None:
            return cached
        if self._sources:
            self.prepare()
            cached = self.cache_policy.get(split)
            if cached is not None:
                return cached
        return []

__all__ = ["DATA_DIR", "CORPUS_DIR", "IDataset", "Dataset", "DataSplits"]
