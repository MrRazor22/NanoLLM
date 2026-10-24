from collections import defaultdict
import logging
from typing import Optional, Protocol, Sequence, runtime_checkable

from data.case import CaseMeta, DecisionCase
from data.policies.balancer import ICaseBalancer, RoundRobinCaseBalancer
from data.policies.census import CategoryBucket, DatasetCensus, ICensusPolicy, CaseCensusPolicy
from data.policies.leakage import ILeakagePolicy, JaccardLeakagePolicy
from data.sources.source import IDataSource, MultiSource
from data.storage import DiskStoragePolicy, IStoragePolicy

logger = logging.getLogger(__name__)

@runtime_checkable
class IDataset(Protocol):
    def prepare(self, source: Optional[IDataSource] = None) -> Sequence[DecisionCase]: ...
    def load(self, split: Optional[str] = None) -> Sequence[DecisionCase]: ...

class Dataset(IDataset):
    def __init__(
        self,
        source: Optional[IDataSource] = None,
        census_policy: Optional[ICensusPolicy] = None,
        balancer_policy: Optional[ICaseBalancer] = None,
        leakage_policy: Optional[ILeakagePolicy] = None,
        storage_policy: Optional[IStoragePolicy] = None,
        train_quota_per_category: int = 1500,
        val_quota_per_category: int = 150,
        buffer_multiplier: int = 2,
        chunk_size: int = 3000,
        seed: int = 42,
    ):
        self.source = source or MultiSource(chunk_size=chunk_size)
        self.census_policy = census_policy or CaseCensusPolicy()
        self.balancer_policy = balancer_policy or RoundRobinCaseBalancer(seed=seed)
        self.leakage_policy = leakage_policy or JaccardLeakagePolicy()
        self.storage_policy = storage_policy or DiskStoragePolicy()
        self.train_quota_per_category = train_quota_per_category
        self.val_quota_per_category = val_quota_per_category
        self.buffer_multiplier = buffer_multiplier
        self.chunk_size = chunk_size
        self.seed = seed

    def prepare(self, source: Optional[IDataSource] = None) -> Sequence[DecisionCase]:
        src = source or self.source
        category_quota = self.train_quota_per_category + self.val_quota_per_category
        buffer_capacity = category_quota * self.buffer_multiplier

        census = DatasetCensus()
        test_cases: list[DecisionCase] = []

        for chunk in src.stream():
            census = census.union(self.census_policy.create(chunk))
            test_cases.extend(c for c in chunk if c.meta.split == "test")

        buffers_by_category: dict[str, list[DecisionCase]] = defaultdict(list)
        completed_categories: set[str] = set()
        balanced_by_category: dict[str, Sequence[DecisionCase]] = {}

        for chunk in src.stream():
            for c in chunk:
                if c.meta.split == "test":
                    continue
                cat = c.meta.category
                if cat in completed_categories:
                    continue

                buffers_by_category[cat].append(c)
                if len(buffers_by_category[cat]) >= buffer_capacity:
                    cat_bucket = census.categories.get(cat)
                    if cat_bucket:
                        clean = self.leakage_policy.decontaminate(buffers_by_category[cat], census.holdouts)
                        balanced = self.balancer_policy.balance(clean, category_bucket=cat_bucket, limit=category_quota)
                        if len(balanced) >= category_quota:
                            balanced_by_category[cat] = balanced[:category_quota]
                            completed_categories.add(cat)
                            del buffers_by_category[cat]

            if len(completed_categories) == len(census.categories) and census.categories:
                break

        for cat, buf in buffers_by_category.items():
            if cat not in completed_categories and buf:
                cat_bucket = census.categories.get(cat)
                if cat_bucket:
                    clean = self.leakage_policy.decontaminate(buf, census.holdouts)
                    balanced = self.balancer_policy.balance(clean, category_bucket=cat_bucket, limit=category_quota)
                    balanced_by_category[cat] = balanced[:category_quota]

        prepared_train: list[DecisionCase] = []
        prepared_val: list[DecisionCase] = []

        for cat, cases in balanced_by_category.items():
            train_count = min(len(cases), self.train_quota_per_category)
            val_count = min(len(cases) - train_count, self.val_quota_per_category)
            prepared_train.extend(c.with_split("train") for c in cases[:train_count])
            prepared_val.extend(c.with_split("val") for c in cases[train_count : train_count + val_count])

        all_cases = [*test_cases, *prepared_train, *prepared_val]
        self.storage_policy.save(all_cases)
        logger.info(
            f"Dataset prepared: {len(test_cases)} test holdouts, {len(prepared_train)} train, {len(prepared_val)} val."
        )
        return all_cases

    def load(self, split: Optional[str] = None) -> Sequence[DecisionCase]:
        return self.storage_policy.load(split=split)

__all__ = ["IDataset", "Dataset"]
