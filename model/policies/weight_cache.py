from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Optional, Protocol, Sequence, runtime_checkable

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DEFAULT_WEIGHTS_PATH = DEFAULT_WEIGHTS_DIR / "piqo.safetensors"

@dataclass(frozen=True, slots=True)
class WeightMeta:
    epoch: int = 1
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    saved_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

@dataclass(frozen=True, slots=True)
class Weights:
    payload: bytes
    name: str = "piqo"
    meta: WeightMeta = field(default_factory=WeightMeta)

    @property
    def size_mb(self) -> float:
        return round(len(self.payload) / (1024 * 1024), 1)

@runtime_checkable
class IWeightCache(Protocol):
    def get(self, path: Path) -> Optional[Weights]: ...
    def put(self, path: Path, weights: Weights) -> Path: ...
    def list(self) -> Sequence[Weights]: ...

class DiskWeightCache(IWeightCache):
    def __init__(self, root: Optional[Path] = None):
        self.root = root or DEFAULT_WEIGHTS_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: Path) -> tuple[Path, Path]:
        base = self.root / path.stem if path.parent == Path(".") else path.parent / path.stem
        return base.with_suffix(".safetensors"), base.with_suffix(".json")

    def get(self, path: Path) -> Optional[Weights]:
        w_file, m_file = self._resolve(path)
        if not w_file.exists():
            return None
        meta_dict = json.loads(m_file.read_text(encoding="utf-8")) if m_file.exists() else {}
        return Weights(
            payload=w_file.read_bytes(),
            name=meta_dict.get("name", w_file.stem),
            meta=WeightMeta(
                epoch=meta_dict.get("epoch", 1),
                train_loss=meta_dict.get("train_loss"),
                val_loss=meta_dict.get("val_loss"),
                saved_at=meta_dict.get("saved_at", ""),
            ),
        )

    def put(self, path: Path, weights: Weights) -> Path:
        w_file, m_file = self._resolve(path)
        w_file.parent.mkdir(parents=True, exist_ok=True)
        w_file.write_bytes(weights.payload)
        meta_dict = {
            "name": weights.name,
            "epoch": weights.meta.epoch,
            "train_loss": weights.meta.train_loss,
            "val_loss": weights.meta.val_loss,
            "saved_at": weights.meta.saved_at,
        }
        m_file.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")
        return w_file

    def list(self) -> Sequence[Weights]:
        if not self.root.exists():
            return []
        results = []
        for w_file in self.root.glob("*.safetensors"):
            m_file = w_file.with_suffix(".json")
            meta_dict = json.loads(m_file.read_text(encoding="utf-8")) if m_file.exists() else {}
            results.append(
                Weights(
                    payload=b"",
                    name=meta_dict.get("name", w_file.stem),
                    meta=WeightMeta(
                        epoch=meta_dict.get("epoch", 1),
                        train_loss=meta_dict.get("train_loss"),
                        val_loss=meta_dict.get("val_loss"),
                        saved_at=meta_dict.get("saved_at", ""),
                    ),
                )
            )
        return sorted(results, key=lambda x: x.meta.saved_at, reverse=True)
