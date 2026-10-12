from data.case import IDataSource, SourceRecord, SplitTag
from data.sources.decisions_source import DecisionsSource
from data.sources.glaive_source import GlaiveSource
from data.sources.huggingface_source import (
    RAW_DIR,
    BenchmarkSpec,
    HuggingFaceSource,
    STANDARD_BENCHMARKS,
    read_raw_jsonl,
)

def standard_sources(seed: int = 42) -> list[IDataSource]:
    """Default collection of diverse multi-task decision sources with unified train/test options."""
    return [
        GlaiveSource(seed=seed),
        DecisionsSource(repeat=5),
        *(HuggingFaceSource(spec) for spec in STANDARD_BENCHMARKS),
    ]

__all__ = [
    "IDataSource",
    "SourceRecord",
    "SplitTag",
    "RAW_DIR",
    "read_raw_jsonl",
    "GlaiveSource",
    "DecisionsSource",
    "HuggingFaceSource",
    "BenchmarkSpec",
    "STANDARD_BENCHMARKS",
    "standard_sources",
]
