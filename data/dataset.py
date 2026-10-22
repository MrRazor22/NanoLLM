import logging
from typing import List, Optional, Protocol, Sequence, runtime_checkable

from data.case import DataSplits, DecisionCase
from data.storage import DiskStoragePolicy, IStoragePolicy
from data.split.split import BudgetSplitPolicy, ISplitPolicy
from data.sources.source import IDataSource, MultiSource

logger = logging.getLogger(__name__)

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract: delivers partitioned dataset splits."""
    def prepare(self) -> DataSplits: ...
    def load(self) -> DataSplits: ...

def default_corpus_source() -> IDataSource:
    """Assembles the default multi-benchmark training source."""
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
    """Sovereign Core Primitive: manages split preparation and persistence of decision cases."""

    def __init__(
        self,
        source: Optional[IDataSource] = None,
        split_policy: Optional[ISplitPolicy] = None,
        storage_policy: Optional[IStoragePolicy] = None,
    ):
        self.source = source or default_corpus_source()
        self.split_policy = split_policy or BudgetSplitPolicy()
        self.storage = storage_policy or DiskStoragePolicy()
        self._splits: Optional[DataSplits] = None

    def prepare(self) -> DataSplits:
        raw_cases = self.source.cases()
        if not raw_cases:
            raise ValueError("No decision cases available for dataset preparation.")
        splits = self.split_policy.partition(raw_cases)
        self._splits = splits
        self.storage.save(splits)
        return splits

    def load(self) -> DataSplits:
        if self._splits is not None:
            return self._splits
        try:
            self._splits = self.storage.load()
            return self._splits
        except FileNotFoundError:
            return self.prepare()

__all__ = ["IDataset", "Dataset", "default_corpus_source"]
