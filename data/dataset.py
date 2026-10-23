import logging
from typing import Optional, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase
from data.storage import DiskStoragePolicy, IStoragePolicy
from data.split.split import BudgetSplitPolicy, ISplitPolicy
from data.sources.source import IDataSource, MultiSource

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
        split_policy: Optional[ISplitPolicy] = None,
        storage_policy: Optional[IStoragePolicy] = None,
    ):
        self.source = source or default_corpus_source()
        self.split_policy = split_policy or BudgetSplitPolicy()
        self.storage = storage_policy or DiskStoragePolicy()

    def prepare(self) -> Sequence[DecisionCase]:
        raw_cases = self.source.cases()
        if not raw_cases:
            raise ValueError("No decision cases available for dataset preparation.")
        cases = self.split_policy.partition(raw_cases)
        self.storage.save(cases)
        return cases

    def load(self, split: Optional[str] = None) -> Sequence[DecisionCase]:
        try:
            return self.storage.load(split)
        except FileNotFoundError:
            all_cases = self.prepare()
            return [c for c in all_cases if c.meta and c.meta.split == split] if split is not None else all_cases

__all__ = ["IDataset", "Dataset", "default_corpus_source"]
