from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable
from inference.questions import DecisionSample

@dataclass(frozen=True)
class SourceRecord:
    """Pure domain DTO tagging sample with dataset origin split."""
    sample: DecisionSample
    tag: str = "unsplit"  # "train", "test", or "unsplit"

@runtime_checkable
class IDataSource(Protocol):
    """Sovereign contract for raw data extractors and adapters."""
    name: str
    def read(self) -> Sequence[SourceRecord]: ...

__all__ = ["SourceRecord", "IDataSource"]
