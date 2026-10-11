"""Data Domain: Sovereign Dataset Preparation, Anti-Leakage, and Collation (PFD)."""

from data.dataset import (
    DATA_DIR,
    CORPUS_DIR,
    IDataset,
    Dataset,
    DataSplits,
)
from data.policies.cache import ICachePolicy, DiskCachePolicy
from data.policies.leakage import ILeakagePolicy, NGramLeakagePolicy
from data.policies.split import ISplitPolicy, BudgetSplitPolicy
from data.sources import (
    RAW_DIR,
    ADAPTED_DIR,
    IDataSource,
    SourceRecord,
    DecisionsSource,
    GenericSource,
    GlaiveSource,
    standard_sources,
)

__all__ = [
    # Core Primitives & DTOs
    "IDataset",
    "Dataset",
    "DataSplits",
    "SourceRecord",
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
    "GenericSource",
    "standard_sources",
    # Directories
    "DATA_DIR",
    "CORPUS_DIR",
    "RAW_DIR",
    "ADAPTED_DIR",
]
