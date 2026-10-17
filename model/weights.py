from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

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

def list_weights(root: Optional[Path] = None) -> Sequence[Weights]:
    """Domain Operation: Discovers saved weights and their metadata from weights directory."""
    weights_dir = root or DEFAULT_WEIGHTS_DIR
    if not weights_dir.exists():
        return []
    files = {f.stem: f for f in weights_dir.glob("*.safetensors")}
    results = []
    for stem, w_file in files.items():
        meta_file = w_file.with_suffix(".json")
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        results.append(
            Weights(
                name=meta.get("name", stem),
                size_mb=round(w_file.stat().st_size / (1024 * 1024), 1),
                meta=WeightMeta(
                    epoch=meta.get("epoch", 1),
                    train_loss=meta.get("train_loss"),
                    val_loss=meta.get("val_loss"),
                    saved_at=meta.get("saved_at"),
                ),
            )
        )
    return sorted(results, key=lambda x: x.meta.saved_at or "", reverse=True)

def has_weights(name: str = "piqo", root: Optional[Path] = None) -> bool:
    """Domain Operation: Verifies whether a weight checkpoint is available in the cache."""
    return any(w.name == name for w in list_weights(root))
