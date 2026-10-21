from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Sequence
from safetensors import safe_open
from model.torch_model import DEFAULT_WEIGHTS_DIR

@dataclass(frozen=True, slots=True)
class WeightInfo:
    name: str
    path: Path
    size_mb: float
    meta: dict[str, str]

def list_weights(root: Optional[Path] = None) -> Sequence[WeightInfo]:
    weights_dir = root or DEFAULT_WEIGHTS_DIR
    if not weights_dir.exists():
        return []
    results = []
    for f in weights_dir.glob("*.safetensors"):
        meta = {}
        try:
            with safe_open(str(f), framework="pt") as handle:
                meta = handle.metadata() or {}
        except Exception:
            pass
        results.append(
            WeightInfo(
                name=f.stem,
                path=f,
                size_mb=round(f.stat().st_size / (1024 * 1024), 1),
                meta=meta,
            )
        )
    return sorted(results, key=lambda x: x.meta.get("saved_at", ""), reverse=True)
