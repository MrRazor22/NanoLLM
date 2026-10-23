import logging
from typing import List, Optional, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class IDataSource(Protocol):
    def cases(self) -> Sequence[DecisionCase]: ...

class MultiSource(IDataSource):
    def __init__(self, sources: Optional[Sequence[IDataSource]] = None):
        if sources is not None:
            self.sources = list(sources)
        else:
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

            self.sources = [
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
            ]

    def cases(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for src in self.sources:
            src_cases = src.cases()
            logger.info("Ingested %s: %d cases", type(src).__name__, len(src_cases))
            cases.extend(src_cases)
        logger.info("Consolidated %d decision cases across %d sources", len(cases), len(self.sources))
        return cases

__all__ = ["IDataSource", "MultiSource"]
