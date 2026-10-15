import orjson
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

    def get(self, target: Union[str, Path]) -> Optional[Weights]:
        p = Path(target)
        w_file = p if p.is_file() else (self.root_dir / p.name if (self.root_dir / p.name).is_file() else (self.root_dir / p.stem).with_suffix(".safetensors"))
        if not w_file.is_file():
            return None
        meta_file = w_file.with_suffix(".json")
        meta = orjson.loads(meta_file.read_bytes()) if meta_file.exists() else {}
        return Weights.from_bytes(
            payload=w_file.read_bytes(),
            name=meta.get("name", w_file.stem),
            epoch=meta.get("epoch", 1),
            train_loss=meta.get("train_loss"),
            val_loss=meta.get("val_loss"),
            saved_at=meta.get("saved_at"),
        )

    def put(self, target: Union[str, Path], weights: Weights) -> Path:
        p = Path(target)
        w_file = (p if p.is_absolute() or p.parent != Path(".") else self.root_dir / p.stem).with_suffix(".safetensors")
        w_file.parent.mkdir(parents=True, exist_ok=True)
        if weights.payload:
            w_file.write_bytes(weights.payload)

        meta = {
            "name": weights.name or w_file.stem,
            "epoch": weights.epoch,
            "train_loss": weights.train_loss,
            "val_loss": weights.val_loss,
            "saved_at": weights.saved_at,
        }
        w_file.with_suffix(".json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
        return w_file
