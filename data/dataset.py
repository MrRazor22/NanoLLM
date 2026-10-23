from dataclasses import replace
import logging
import random
from typing import Optional, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase
from data.policies.balancer import ICaseBalancer, RoundRobinCaseBalancer
from data.sources.source import IDataSource, MultiSource
from data.storage import DiskStoragePolicy, IStoragePolicy

logger = logging.getLogger(__name__)

@runtime_checkable
class IDataset(Protocol):
    def prepare(self) -> Sequence[DecisionCase]: ...
    def load(self, split: Optional[str] = None) -> Sequence[DecisionCase]: ...

def default_corpus_source() -> IDataSource:
    from data.sources.anli_source import ANLISource
    from data.sources.art_source import ARTSource
    from data.sources.copa_source import COPASource
    from data.sources.typed_decisions_source import TypedDecisionsSource
    from data.sources.glaive_source import GlaiveSource
    from data.sources.hellaswag_source import HellaSwagSource
    from data.sources.classification_source import ClassificationSource, STANDARD_CLASSIFICATIONS
    from data.sources.medqa_source import MedQASource
    from data.sources.multichoice_source import MultiChoiceSource, STANDARD_MULTICHOICE
    from data.sources.sciq_source import SciQSource
    from data.sources.reclor_source import ReClorSource

    return MultiSource([
        GlaiveSource(),
        TypedDecisionsSource(),
        ANLISource(),
        HellaSwagSource(),
        MedQASource(),
        ARTSource(),
        COPASource(),
        SciQSource(),
        ReClorSource(),
        *(ClassificationSource(s) for s in STANDARD_CLASSIFICATIONS),
        *(MultiChoiceSource(s) for s in STANDARD_MULTICHOICE),
    ])

class Dataset(IDataset):
    def __init__(
        self,
        source: Optional[IDataSource] = None,
        balancer: Optional[ICaseBalancer] = None,
        storage: Optional[IStoragePolicy] = None,
        max_train: int = 1500,
        max_val: int = 150,
        max_test: int = 400,
        seed: int = 42,
    ):
        self.source = source or default_corpus_source()
        self.balancer = balancer or RoundRobinCaseBalancer()
        self.storage = storage or DiskStoragePolicy()
        self.max_train = max_train
        self.max_val = max_val
        self.max_test = max_test
        self.seed = seed

    def prepare(self) -> Sequence[DecisionCase]:
        raw_cases = self.source.cases()
        if not raw_cases:
            raise ValueError("No decision cases available for dataset preparation.")

        rng = random.Random(self.seed)
        by_cat: dict[str, list[DecisionCase]] = {}
        for c in raw_cases:
            if not c.meta:
                raise ValueError("DecisionCase missing CaseMeta.")
            by_cat.setdefault(c.meta.category, []).append(c)

        all_cases: list[DecisionCase] = []
        for cat, items in by_cat.items():
            pre_test = [c for c in items if c.meta.split == "test"]
            pool = [c for c in items if c.meta.split != "test"]

            cat_test = list(self.balancer.balance(pre_test if pre_test else pool, self.max_test, rng)) if (pre_test or pool) else []
            test_ids = {id(c) for c in cat_test}
            pool = [c for c in pool if id(c) not in test_ids]

            cat_val = list(self.balancer.balance(pool, self.max_val, rng, holdouts=cat_test)) if pool else []
            val_ids = {id(c) for c in cat_val}
            pool = [c for c in pool if id(c) not in val_ids]

            cat_train = list(self.balancer.balance(pool, self.max_train, rng, holdouts=cat_test)) if pool else []

            all_cases.extend(replace(c, meta=replace(c.meta, split="test")) for c in cat_test)
            all_cases.extend(replace(c, meta=replace(c.meta, split="val")) for c in cat_val)
            all_cases.extend(replace(c, meta=replace(c.meta, split="train")) for c in cat_train)

        self.storage.save(all_cases)
        return all_cases

    def load(self, split: Optional[str] = None) -> Sequence[DecisionCase]:
        try:
            return self.storage.load(split)
        except FileNotFoundError:
            all_cases = self.prepare()
            return [c for c in all_cases if c.meta and c.meta.split == split] if split is not None else all_cases

__all__ = ["IDataset", "Dataset", "default_corpus_source"]
