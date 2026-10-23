import logging
from typing import Iterator, Optional, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class IDataSource(Protocol):
    def stream(self) -> Iterator[Sequence[DecisionCase]]: ...

class MultiSource(IDataSource):
    def __init__(
        self,
        sources: Optional[Sequence[IDataSource]] = None,
        chunk_size: int = 3000,
    ):
        self.chunk_size = chunk_size
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

    def stream(self) -> Iterator[Sequence[DecisionCase]]:
        buffer: list[DecisionCase] = []
        for src in self.sources:
            for chunk in src.stream():
                buffer.extend(chunk)
                while len(buffer) >= self.chunk_size:
                    yield tuple(buffer[:self.chunk_size])
                    buffer = buffer[self.chunk_size:]
        if buffer:
            yield tuple(buffer)

__all__ = ["IDataSource", "MultiSource"]
