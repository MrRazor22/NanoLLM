from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True, slots=True)
class Weights:
    """Core Domain DTO: Substrate-neutral parameter payload and training metadata."""
    payload: bytes = b""
    name: str = "piqo"
    epoch: int = 1
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    saved_at: Optional[str] = None
    size_mb: float = 0.0

    def to_bytes(self) -> bytes:
        """Pure Transform: returns raw parameter byte payload."""
        return self.payload

    @classmethod
    def from_bytes(cls, payload: bytes, **kwargs) -> "Weights":
        """Pure Transform: constructs Weights DTO from raw byte payload."""
        size_mb = kwargs.pop("size_mb", round(len(payload) / (1024 * 1024), 1))
        return cls(payload=payload, size_mb=size_mb, **kwargs)
