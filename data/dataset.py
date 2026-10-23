from collections import defaultdict
import logging
from typing import Optional, Protocol, Sequence, runtime_checkable

from data.case import CaseMeta, DecisionCase
from data.policies.balancer import ICaseBalancer, RoundRobinCaseBalancer
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
        leakage_policy: Optional[ILeakagePolicy] = None,
        balancer_policy: Optional[ICaseBalancer] = None,
        storage_policy: Optional[IStoragePolicy] = None,
        train_quota_per_category: int = 1500,
        val_quota_per_category: int = 150,
        chunk_size_per_category: int = 3000,
    ):
        self.source = source or MultiSource()
        self.leakage_policy = leakage_policy or JaccardLeakagePolicy()
        self.balancer_policy = balancer_policy or RoundRobinCaseBalancer()
        self.storage_policy = storage_policy or DiskStoragePolicy()
        self.train_quota_per_category = train_quota_per_category
        self.val_quota_per_category = val_quota_per_category
        self.chunk_size_per_category = chunk_size_per_category

    def prepare(self, source: Optional[IDataSource] = None) -> Sequence[DecisionCase]:
        src = source or self.source
        cases = src.cases()
        if not cases:
            raise ValueError("No decision cases available for dataset preparation.")

        train_by_category, test_by_category = defaultdict(list), defaultdict(list)
        for c in cases:
            (test_by_category if c.meta.split == "test" else train_by_category)[c.meta.category].append(c)

        prepared_train: list[DecisionCase] = []
        prepared_val: list[DecisionCase] = []
        category_quota = self.train_quota_per_category + self.val_quota_per_category

        for cat, cat_train_pool in train_by_category.items():
            cat_test_cases = test_by_category.get(cat, [])
            self.balancer_policy.prepare(cat_train_pool)

            category_cases: list[DecisionCase] = []
            cursor = 0
            pool_len = len(cat_train_pool)

            while len(category_cases) < category_quota and cursor < pool_len:
                raw_chunk = cat_train_pool[cursor : cursor + self.chunk_size_per_category]
                cursor += self.chunk_size_per_category

                clean_chunk = (
                    self.leakage_policy.decontaminate(raw_chunk, cat_test_cases)
                    if cat_test_cases
                    else raw_chunk
                )
                balanced_chunk = self.balancer_policy.balance(clean_chunk)
                category_cases.extend(balanced_chunk)

            train_count = min(len(category_cases), self.train_quota_per_category)
            val_count = min(len(category_cases) - train_count, self.val_quota_per_category)

            prepared_train.extend(c.with_split("train") for c in category_cases[:train_count])
            prepared_val.extend(c.with_split("val") for c in category_cases[train_count : train_count + val_count])

        test_cases = [c for pool in test_by_category.values() for c in pool]
        all_cases = [*test_cases, *prepared_train, *prepared_val]
        self.storage_policy.save(all_cases)
        logger.info(
            f"Dataset prepared: {len(test_cases)} test holdouts, {len(prepared_train)} train, {len(prepared_val)} val."
        )
        return all_cases

    def load(self, split: Optional[str] = None) -> Sequence[DecisionCase]:
        return self.storage_policy.load(split=split)
