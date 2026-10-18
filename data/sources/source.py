from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase

@runtime_checkable
class IDataSource(Protocol):
    """Sovereign policy contract: yields domain records from external data origins."""
    def cases(self) -> Sequence[DecisionCase]: ...

def get_available_sources() -> Sequence[IDataSource]:
    """Lazy catalog of all available decision sources."""
    from data.sources.anli_source import ANLISource
    from data.sources.art_source import ARTSource
    from data.sources.copa_source import COPASource
    from data.sources.decisions_source import DecisionsSource
    from data.sources.glaive_source import GlaiveSource
    from data.sources.hellaswag_source import HellaSwagSource
    from data.sources.huggingface_source import HuggingFaceSource, STANDARD_BENCHMARKS
    from data.sources.medqa_source import MedQASource
    from data.sources.multichoice_source import MultiChoiceReasoningSource, MULTICHOICE_REASONING_BENCHMARKS
    from data.sources.sciq_source import SciQSource
    from data.sources.reclor_source import ReClorSource

    return [
        GlaiveSource(),
        DecisionsSource(),
        ANLISource(),
        HellaSwagSource(),
        MedQASource(),
        ARTSource(),
        COPASource(),
        SciQSource(),
        ReClorSource(),
        *(HuggingFaceSource(s) for s in STANDARD_BENCHMARKS),
        *(MultiChoiceReasoningSource(s) for s in MULTICHOICE_REASONING_BENCHMARKS),
    ]

__all__ = ["IDataSource", "get_available_sources"]
