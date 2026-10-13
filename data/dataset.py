from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Union, runtime_checkable

from data.case import DataSplits, DecisionCase, IDataSource, SourceRecord, SplitTag
from data.curation.budget_split import BudgetSplitPolicy, ISplitPolicy
from data.curation.ngram_leakage import ILeakagePolicy, NGramLeakagePolicy
from data.disk_cache import DiskCachePolicy, ICachePolicy
from data.sources.decisions_source import DecisionsSource
from data.sources.glaive_source import GlaiveSource
from data.sources.huggingface_source import (
    BenchmarkSpec,
    HuggingFaceSource,
    STANDARD_BENCHMARKS,
)
from data.sources.multichoice_source import (
    MultiChoiceReasoningSource,
    MULTICHOICE_REASONING_BENCHMARKS,
)
from data.sources.anli_source import ANLISource

def standard_sources(seed: int = 42) -> list[IDataSource]:
    """Default collection of diverse multi-task decision and reasoning sources with unified train/test options."""
    return [
        GlaiveSource(seed=seed),
        DecisionsSource(repeat=5),
        *(HuggingFaceSource(spec) for spec in STANDARD_BENCHMARKS),
        *(MultiChoiceReasoningSource(spec) for spec in MULTICHOICE_REASONING_BENCHMARKS),
        ANLISource(),
    ]

DATA_DIR = Path(__file__).resolve().parent
CORPUS_DIR = DATA_DIR / "corpus"

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract: manages dataset source preparation, leakage audit, and cached loading."""
    def prepare(self) -> DataSplits: ...
    def load(self, split: SplitTag = "train") -> Sequence[DecisionCase]: ...

class Dataset(IDataset):
    """Sovereign Core Primitive: manages source registration, split preparation, zero leakage, and cached loading."""

    def __init__(
        self,
        split_policy: Optional[ISplitPolicy] = None,
        leakage_policy: Optional[ILeakagePolicy] = None,
        cache_policy: Optional[ICachePolicy] = None,
        sources: Optional[Sequence[IDataSource]] = None,
        corpus_dir: Optional[Union[str, Path]] = None,
    ):
        self.corpus_dir = Path(corpus_dir) if corpus_dir is not None else CORPUS_DIR
        self.split_policy = split_policy or BudgetSplitPolicy()
        self.leakage_policy = leakage_policy or NGramLeakagePolicy()
        self.cache_policy = cache_policy or DiskCachePolicy(root_dir=self.corpus_dir)
        self.sources: List[IDataSource] = list(sources) if sources is not None else standard_sources()
        self.leak_audit: dict[str, int] = {}

    @property
    def signature(self) -> str:
        import hashlib
        import json
        payload = {
            "version": 1,
            "split_policy": self.split_policy.signature,
            "leakage_policy": self.leakage_policy.signature,
            "cache_policy": self.cache_policy.signature,
            "sources": [s.signature for s in self.sources],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def prepare(self) -> DataSplits:
        """Extracts all registered sources, partitions splits, audits for leakage, and caches via cache policy."""
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
        if unique_val_test:
            splits = DataSplits(
                train=splits.train,
                val=[c for c in splits.val if id(c) not in unique_val_test],
                test=splits.test,
            )

        unique_train_leaks = {id(c): c for c in (test_leaks + val_leaks)}
        if unique_train_leaks:
            splits = DataSplits(
                train=[c for c in splits.train if id(c) not in unique_train_leaks],
                val=splits.val,
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

        # Delegate caching to injected cache policy with signature
        sig = self.signature
        self.cache_policy.put("train", splits.train, signature=sig)
        self.cache_policy.put("val", splits.val, signature=sig)
        self.cache_policy.put("test", splits.test, signature=sig)

        return splits

    def load(self, split: SplitTag = "train") -> Sequence[DecisionCase]:
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
