"""Data source registry for available decision and reasoning sources (PFD)."""

from data.sources.source import IDataSource
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

AVAILABLE_SOURCES = {
    "glaive": GlaiveSource,
    "decisions": DecisionsSource,
    "anli": ANLISource,
    "hellaswag": HellaSwagSource,
    "medqa": MedQASource,
    "art": ARTSource,
    "copa": COPASource,
    "sciq": SciQSource,
    "reclor": ReClorSource,
    **{s.category: (lambda spec=s: HuggingFaceSource(spec)) for s in STANDARD_BENCHMARKS},
    **{s.category: (lambda spec=s: MultiChoiceReasoningSource(spec)) for s in MULTICHOICE_REASONING_BENCHMARKS},
}

__all__ = ["AVAILABLE_SOURCES"]
