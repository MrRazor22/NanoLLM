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
        quota_per_category: int = 1500,
        val_ratio: float = 0.10,
        chunk_factor: int = 2,
    ):
        self.source = source or MultiSource()
        self.leakage_policy = leakage_policy or JaccardLeakagePolicy()
        self.balancer_policy = balancer_policy or RoundRobinCaseBalancer()
        self.storage_policy = storage_policy or DiskStoragePolicy()
        self.quota_per_category = quota_per_category
        self.val_ratio = val_ratio
        self.chunk_factor = chunk_factor

    def prepare(self, source: Optional[IDataSource] = None) -> Sequence[DecisionCase]:
        src = source or self.source
        cases = src.cases()
        if not cases:
            raise ValueError("No decision cases available for dataset preparation.")

        holdouts: list[DecisionCase] = []
        pool_by_category: dict[str, list[DecisionCase]] = defaultdict(list)
        holdouts_by_category: dict[str, list[DecisionCase]] = defaultdict(list)

        for c in cases:
            if not c.meta:
                continue
            cat = c.meta.category
            if c.meta.split == "test":
                holdouts.append(c)
                holdouts_by_category[cat].append(c)
            else:
                pool_by_category[cat].append(c)

        train_cases: list[DecisionCase] = []
        for cat, cat_pool in pool_by_category.items():
            cat_holdouts = holdouts_by_category.get(cat, [])
            clean_collected: list[DecisionCase] = []
            cursor = 0
            pool_len = len(cat_pool)

            while len(clean_collected) < self.quota_per_category and cursor < pool_len:
                needed = self.quota_per_category - len(clean_collected)
                chunk_size = needed * self.chunk_factor
                raw_chunk = cat_pool[cursor : cursor + chunk_size]
                cursor += chunk_size

                balanced_chunk = self.balancer_policy.balance(raw_chunk)
                clean_chunk = (
                    self.leakage_policy.decontaminate(balanced_chunk, cat_holdouts)
                    if cat_holdouts
                    else balanced_chunk
                )
                clean_collected.extend(clean_chunk)

            final_clean = self.balancer_policy.balance(clean_collected, limit=self.quota_per_category)
            train_cases.extend(final_clean)

        val_count = max(1, int(len(train_cases) * self.val_ratio)) if len(train_cases) > 1 else 0
        prepared_val = [c.with_split("val") for c in train_cases[:val_count]]
        prepared_train = [c.with_split("train") for c in train_cases[val_count:]]

        all_cases = [*holdouts, *prepared_train, *prepared_val]
        self.storage_policy.save(all_cases)
        logger.info(
            f"Dataset prepared: {len(holdouts)} test holdouts, {len(prepared_train)} train, {len(prepared_val)} val."
        )
        return all_cases

    def load(self, split: Optional[str] = None) -> Sequence[DecisionCase]:
        return self.storage_policy.load(split=split)
