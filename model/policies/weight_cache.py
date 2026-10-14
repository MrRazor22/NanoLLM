import json
import os
from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable

from model.weights import Weights

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

@runtime_checkable
class IWeightCache(Protocol):
    """Internal opinion sub-primitive contract: local persistence for neural weights."""
    def get(self, target: Union[str, Path]) -> Optional[Weights]: ...
    def put(self, target: Union[str, Path], weights: Weights) -> Path: ...

class DiskWeightCache(IWeightCache):
    """Local filesystem weight cache implementation."""

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_WEIGHTS_DIR

    def _resolve_file(self, target: Union[str, Path]) -> Path:
        p = Path(target)
        if p.is_absolute() or p.parent != Path("."):
            return p if p.suffix == ".pt" else p.with_suffix(".pt")
        s = str(target)
        filename = s if s.endswith(".pt") else f"{s}.pt"
        return self.root_dir / filename

    def get(self, target: Union[str, Path]) -> Optional[Weights]:
        w_file = self._resolve_file(target)
        if not w_file.exists():
            return None
        meta_file = w_file.parent / f"{w_file.stem}.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        return Weights.from_bytes(
            payload=w_file.read_bytes(),
            name=meta.get("name", w_file.name),
            epoch=meta.get("epoch", 1),
            train_loss=meta.get("train_loss"),
            val_loss=meta.get("val_loss"),
            saved_at=meta.get("saved_at"),
        )

    def put(self, target: Union[str, Path], weights: Weights) -> Path:
        w_file = self._resolve_file(target)
        w_file.parent.mkdir(parents=True, exist_ok=True)
        if weights.payload:
            w_file.write_bytes(weights.payload)

        meta_file = w_file.parent / f"{w_file.stem}.json"
        meta = {
            "name": weights.name or w_file.name,
            "epoch": weights.epoch,
            "train_loss": weights.train_loss,
            "val_loss": weights.val_loss,
            "saved_at": weights.saved_at,
        }
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return w_file
