from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union
import torch

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "weights"

def get_latest_checkpoint(prefix: str = "checkpoint_champion") -> Tuple[Optional[Path], int]:
    if not CHECKPOINTS_DIR.exists(): return None, 0
    files = []
    for f in CHECKPOINTS_DIR.glob(f"{prefix}*.pt"):
        m = re.match(rf"^{re.escape(prefix)}(?:_v(\d+))?\.pt$", f.name)
        if m: files.append((int(m.group(1) or 1), f))
    return max(files, key=lambda x: x[0], default=(0, None))[::-1]

def resolve_default_checkpoint() -> Path:
    latest, _ = get_latest_checkpoint()
    for p in [latest, CHECKPOINTS_DIR / "checkpoint_champion_v10.pt", CHECKPOINTS_DIR / "checkpoint_champion_v8.pt"]:
        if p and p.exists(): return p
    return CHECKPOINTS_DIR / "checkpoint_champion.pt"

DEFAULT_CHECKPOINT = resolve_default_checkpoint()

@dataclass(frozen=True)
class CheckpointMeta:
    """Explicit domain DTO for persisted model checkpoint metadata."""
    path: Union[str, Path] = DEFAULT_CHECKPOINT
    epoch: int = 1
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    saved_at: Optional[str] = None

@dataclass(frozen=True)
class CheckpointInfo:
    name: str
    path: Path
    version: int
    size_mb: float
    modified: float
    is_latest: bool
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    step: Optional[int] = None
    epoch: Optional[int] = None
    saved_at: Optional[str] = None

def save_checkpoint(
    model: object,
    meta: Optional[CheckpointMeta] = None,
) -> None:
    """Persists model state dictionary and metadata to disk."""
    m = meta or CheckpointMeta()
    p = Path(m.path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta_dict = {
        "epoch": m.epoch,
        "train_loss": m.train_loss,
        "val_loss": m.val_loss,
        "saved_at": m.saved_at,
    }
    state_dict = model.state_dict() if hasattr(model, "state_dict") else {}
    torch.save({"model_state_dict": state_dict, "meta": meta_dict}, str(p))

def get_next_checkpoint_path(prefix: str = "checkpoint_champion") -> Path:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    _, ver = get_latest_checkpoint(prefix)
    return CHECKPOINTS_DIR / f"{prefix}_v{ver + 1}.pt"

def list_all_checkpoints() -> List[CheckpointInfo]:
    if not CHECKPOINTS_DIR.exists(): return []
    manifest_p = CHECKPOINTS_DIR / "manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {}
    latest, _ = get_latest_checkpoint()
    items = []
    for f in CHECKPOINTS_DIR.glob("*.pt"):
        m = re.match(r"^checkpoint_champion(?:_v(\d+))?\.pt$", f.name)
        meta = manifest.get(f.name, {})
        stat = f.stat()
        items.append(CheckpointInfo(
            name=f.name, path=f, version=int(m.group(1) or 1) if m else 0,
            size_mb=round(stat.st_size / (1024 * 1024), 1), modified=stat.st_mtime,
            is_latest=latest is not None and f.resolve() == latest.resolve(),
            train_loss=meta.get("train_loss"), val_loss=meta.get("val_loss"),
            step=meta.get("step"), epoch=meta.get("epoch"), saved_at=meta.get("saved_at"),
        ))
    return sorted(items, key=lambda x: x.version, reverse=True)

__all__ = [
    "CHECKPOINTS_DIR", "DEFAULT_CHECKPOINT", "CheckpointMeta", "CheckpointInfo",
    "save_checkpoint", "get_latest_checkpoint", "get_next_checkpoint_path",
    "list_all_checkpoints", "resolve_default_checkpoint",
]
