"""Data Domain: Sovereign Dataset Preparation, Anti-Leakage, and Collation (PFD)."""

from data.case import DataSplits, DecisionCase, IDataSource, SourceRecord, SplitTag
from data.dataset import (
    DATA_DIR,
    CORPUS_DIR,
    IDataset,
    Dataset,
)
from data.curation import (
    BudgetSplitPolicy,
    ILeakagePolicy,
    ISplitPolicy,
    NGramLeakagePolicy,
)
from data.disk_cache import DiskCachePolicy, ICachePolicy
from data.sources import (
    RAW_DIR,
    BenchmarkSpec,
    DecisionsSource,
    HuggingFaceSource,
    GlaiveSource,
    STANDARD_BENCHMARKS,
    standard_sources,
)

__all__ = [
    # Core Primitives & DTOs
    "IDataset",
    "Dataset",
    "DataSplits",
    "SourceRecord",
    "DecisionCase",
    "SplitTag",
    "BenchmarkSpec",
    "STANDARD_BENCHMARKS",
    # Policies
    "ISplitPolicy",
    "BudgetSplitPolicy",
    "ILeakagePolicy",
    "NGramLeakagePolicy",
    "ICachePolicy",
    "DiskCachePolicy",
    # Sources
    "IDataSource",
    "GlaiveSource",
    "DecisionsSource",
    "HuggingFaceSource",
    "standard_sources",
    # Directories
    "DATA_DIR",
    "CORPUS_DIR",
    "RAW_DIR",
]
