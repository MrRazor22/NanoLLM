from pathlib import Path
from typing import List
from data.sources.source import IDataSource
from data.sources.decisions_source import DecisionsSource
from data.sources.glaive_source import GlaiveSource
from data.sources.huggingface_source import (
    HuggingFaceSource,
    STANDARD_BENCHMARKS,
)
from data.sources.multichoice_source import (
    MultiChoiceReasoningSource,
    MULTICHOICE_REASONING_BENCHMARKS,
)

DATA_DIR = Path(__file__).resolve().parent
CORPUS_DIR = DATA_DIR / "corpus"

def standard_sources(seed: int = 42) -> List[IDataSource]:
    """Default collection of diverse multi-task decision and reasoning sources with unified train/test options."""
    return [
        GlaiveSource(seed=seed),
        DecisionsSource(repeat=5),
        *(HuggingFaceSource(spec) for spec in STANDARD_BENCHMARKS),
        *(MultiChoiceReasoningSource(spec) for spec in MULTICHOICE_REASONING_BENCHMARKS),
    ]
