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

def weights_from_bytes(payload: bytes, name: str = "piqo", meta: Optional[WeightMeta] = None) -> Weights:
    """Pure Transform: constructs Weights DTO from raw byte payload and metadata."""
    size_mb = round(len(payload) / (1024 * 1024), 1) if payload else 0.0
    return Weights(payload=payload, name=name, meta=meta or WeightMeta(), size_mb=size_mb)
