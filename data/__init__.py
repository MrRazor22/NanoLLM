"""Data Domain: Sovereign Dataset Preparation, Anti-Leakage, and Collation (PFD)."""

from data.dataset import (
    DATA_DIR,
    CORPUS_DIR,
    IDataset,
    Dataset,
    DataSplits,
    load_raw_jsonl,
    save_jsonl,
    to_decision_sample,
)
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
    # Sources
    "IDataSource",
    "GlaiveSource",
    "DecisionsSource",
    "GenericSource",
    # Utilities
    "DATA_DIR",
    "CORPUS_DIR",
    "RAW_DIR",
    "ADAPTED_DIR",
    "to_decision_sample",
    "load_raw_jsonl",
    "save_jsonl",
]
