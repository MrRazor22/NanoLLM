from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True, slots=True)
class WeightMeta:
    """Domain DTO: Training checkpoint telemetry and provenance metadata."""
    epoch: int = 1
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    saved_at: Optional[str] = None

@dataclass(frozen=True, slots=True)
class Weights:
    """Core Domain DTO: Substrate-neutral parameter payload and metadata."""
    payload: bytes = b""
    name: str = "piqo"
    meta: WeightMeta = field(default_factory=WeightMeta)
    size_mb: float = 0.0

    @property
    def epoch(self) -> int:
        return self.meta.epoch

    @property
    def train_loss(self) -> Optional[float]:
        return self.meta.train_loss

    @property
    def val_loss(self) -> Optional[float]:
        return self.meta.val_loss

    @property
    def saved_at(self) -> Optional[str]:
        return self.meta.saved_at

    def to_bytes(self) -> bytes:
        """Pure Transform: returns raw parameter byte payload."""
        return self.payload

    @classmethod
    def from_bytes(cls, payload: bytes, **kwargs) -> "Weights":
        """Pure Transform: constructs Weights DTO from raw byte payload."""
        size_mb = kwargs.pop("size_mb", round(len(payload) / (1024 * 1024), 1))
        name = kwargs.pop("name", "piqo")
        meta = WeightMeta(
            epoch=kwargs.pop("epoch", 1),
            train_loss=kwargs.pop("train_loss", None),
            val_loss=kwargs.pop("val_loss", None),
            saved_at=kwargs.pop("saved_at", None),
        )
        return cls(payload=payload, name=name, meta=meta, size_mb=size_mb)
