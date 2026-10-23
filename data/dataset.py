import logging
from typing import Optional, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase
from data.policies.balancer import ICaseBalancer, RoundRobinCaseBalancer
from data.policies.leakage import ILeakagePolicy, JaccardLeakagePolicy
from data.policies.split import ISplitPolicy, RatioSplitPolicy
from data.sources.source import IDataSource, MultiSource
from data.storage import DiskStoragePolicy, IStoragePolicy

logger = logging.getLogger(__name__)

@runtime_checkable
class IDataset(Protocol):
    def prepare(self, source: Optional[IDataSource] = None) -> Sequence[DecisionCase]: ...
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
        split_policy: Optional[ISplitPolicy] = None,
        leakage_policy: Optional[ILeakagePolicy] = None,
        balancer_policy: Optional[ICaseBalancer] = None,
        storage_policy: Optional[IStoragePolicy] = None,
    ):
        self.source = source or default_corpus_source()
        self.split_policy = split_policy or RatioSplitPolicy()
        self.leakage_policy = leakage_policy or JaccardLeakagePolicy()
        self.balancer_policy = balancer_policy or RoundRobinCaseBalancer()
        self.storage_policy = storage_policy or DiskStoragePolicy()

    def prepare(self, source: Optional[IDataSource] = None) -> Sequence[DecisionCase]:
        src = source or self.source
        cases = src.cases()
        if not cases:
            raise ValueError("No decision cases available for dataset preparation.")

        splits = self.split_policy.partition(cases)
        clean = self.leakage_policy.decontaminate(splits)
        balanced = self.balancer_policy.balance(clean)
        self.storage_policy.save(balanced)
        return balanced

    def load(self, split: Optional[str] = None) -> Sequence[DecisionCase]:
        try:
            return self.storage_policy.load(split)
        except FileNotFoundError:
            all_cases = self.prepare()
            return [c for c in all_cases if c.meta and c.meta.split == split] if split is not None else all_cases

__all__ = ["IDataset", "Dataset", "default_corpus_source"]
