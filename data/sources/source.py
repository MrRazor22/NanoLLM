from typing import Protocol, Sequence, runtime_checkable
from data.case import SourceRecord

@runtime_checkable
class IDataSource(Protocol):
    """Sovereign policy contract: extracts domain records from external data origins."""
    name: str
    def read(self) -> Sequence[SourceRecord]: ...
