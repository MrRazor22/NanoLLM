import logging
from typing import List, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class IDataSource(Protocol):
    """Sovereign policy contract: yields domain records from external data origins."""
    def cases(self) -> Sequence[DecisionCase]: ...

class MultiSource(IDataSource):
    """Composite data source combining multiple independent sources."""

    def __init__(self, sources: Sequence[IDataSource]):
        self.sources = list(sources)

    def cases(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for src in self.sources:
            src_cases = src.cases()
            logger.info("Ingested %s: %d cases", getattr(src, "name", type(src).__name__), len(src_cases))
            cases.extend(src_cases)
        logger.info("Consolidated %d decision cases across %d sources", len(cases), len(self.sources))
        return cases

__all__ = ["IDataSource", "MultiSource"]
