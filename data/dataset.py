from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Union, runtime_checkable

from data.case import DecisionCase
from data.policies.cache import DiskCachePolicy, ICachePolicy
from data.policies.leakage import ILeakagePolicy, NGramLeakagePolicy
from data.policies.split import BudgetSplitPolicy, DataSplits, ISplitPolicy
from data.sources import IDataSource, SourceRecord, standard_sources

DATA_DIR = Path(__file__).resolve().parent
CORPUS_DIR = DATA_DIR / "corpus"

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract: manages dataset source preparation, leakage audit, and cached loading."""
    def prepare(self, sources: Optional[Sequence[IDataSource]] = None) -> DataSplits: ...
    def load(self, split: str = "train") -> Sequence[DecisionCase]: ...

class Dataset(IDataset):
    """Sovereign Core Primitive: manages source registration, split preparation, zero leakage, and cached loading."""

    def __init__(
        self,
        corpus_dir: Optional[Union[str, Path]] = None,
        train_quota: int = 40000,
        val_quota: int = 2500,
        test_quota: int = 2500,
        n_gram: int = 13,
        seed: int = 42,
        split_policy: Optional[ISplitPolicy] = None,
        leakage_policy: Optional[ILeakagePolicy] = None,
        cache_policy: Optional[ICachePolicy] = None,
        sources: Optional[Sequence[IDataSource]] = None,
    ):
        self.corpus_dir = Path(corpus_dir) if corpus_dir is not None else CORPUS_DIR
        self.split_policy = split_policy or BudgetSplitPolicy(
            train_quota=train_quota,
            val_quota=val_quota,
            test_quota=test_quota,
            seed=seed,
        )
        self.leakage_policy = leakage_policy or NGramLeakagePolicy(n_gram=n_gram)
        self.cache_policy = cache_policy or DiskCachePolicy(root_dir=self.corpus_dir)
        self.sources: List[IDataSource] = list(sources) if sources is not None else standard_sources(seed=seed)

    @property
    def signature(self) -> str:
        import hashlib
        raw = "|".join(f"{s.name}:{getattr(s, 'limit', '')}:{getattr(s, 'category', '')}" for s in self.sources)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def prepare(self, sources: Optional[Sequence[IDataSource]] = None) -> DataSplits:
        """Extracts all registered sources, partitions splits, audits for leakage, and caches via cache policy."""
        src_list = list(sources) if sources is not None else self.sources
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

        # Delegate caching to injected cache policy with signature
        sig = self.signature
        self.cache_policy.put("train", splits.train, signature=sig)
        self.cache_policy.put("val", splits.val, signature=sig)
        self.cache_policy.put("test", splits.test, signature=sig)

        return splits

    def load(self, split: str = "train") -> Sequence[DecisionCase]:
        """Loads domain decision cases from cache; self-prepares from sources on cache miss or signature mismatch."""
        cached = self.cache_policy.get(split, signature=self.signature)
        if cached is not None:
            return cached
        if self.sources:
            self.prepare()
            cached = self.cache_policy.get(split, signature=self.signature)
            if cached is not None:
                return cached
        raise FileNotFoundError(
            f"Corpus partition '{split}' not found and no data sources available to prepare it."
        )

__all__ = ["DATA_DIR", "CORPUS_DIR", "IDataset", "Dataset", "DataSplits"]
