from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable
from inference.case import DecisionCase

@dataclass(frozen=True)
class SourceRecord:
    """Pure domain DTO tagging case with dataset origin split."""
    sample: DecisionCase
    tag: str = "unsplit"  # "train", "test", or "unsplit"

@runtime_checkable
class IDataSource(Protocol):
    """Sovereign contract for raw data extractors and adapters."""
    name: str
    def read(self) -> Sequence[SourceRecord]: ...

__all__ = ["SourceRecord", "IDataSource"]
