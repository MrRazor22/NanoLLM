from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

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

def get_latest_checkpoint(prefix: str = "checkpoint_champion") -> Tuple[Optional[Path], int]:
    if not CHECKPOINTS_DIR.exists():
        return None, 0
    max_ver = -1
    best_file = None
    pattern = re.compile(rf"^{re.escape(prefix)}(?:_v(\d+))?\.pt$")
    for f in CHECKPOINTS_DIR.glob(f"{prefix}*.pt"):
        m = pattern.match(f.name)
        if m:
            ver = int(m.group(1)) if m.group(1) is not None else 1
            if ver > max_ver:
                max_ver = ver
                best_file = f
    return best_file, max(0, max_ver)

def resolve_default_checkpoint() -> Path:
    latest, _ = get_latest_checkpoint()
    if latest and latest.exists():
        return latest
    fallback = CHECKPOINTS_DIR / "checkpoint_champion_v10.pt"
    if fallback.exists():
        return fallback
    return CHECKPOINTS_DIR / "checkpoint_champion_v8.pt"

DEFAULT_CHECKPOINT = resolve_default_checkpoint()

def get_next_checkpoint_path(prefix: str = "checkpoint_champion") -> Path:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    _, latest_ver = get_latest_checkpoint(prefix)
    return CHECKPOINTS_DIR / f"{prefix}_v{latest_ver + 1}.pt"

def list_all_checkpoints() -> List[CheckpointInfo]:
    if not CHECKPOINTS_DIR.exists():
        return []
    manifest_path = CHECKPOINTS_DIR / "manifest.json"
    manifest: Dict[str, Dict[str, Union[str, float, int]]] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass
    latest_ckpt, _ = get_latest_checkpoint()
    pattern = re.compile(r"^checkpoint_champion(?:_v(\d+))?\.pt$")
    results: List[CheckpointInfo] = []
    for f in CHECKPOINTS_DIR.glob("*.pt"):
        m = pattern.match(f.name)
        ver = int(m.group(1)) if (m and m.group(1) is not None) else (1 if m else 0)
        is_latest = latest_ckpt is not None and f.resolve() == latest_ckpt.resolve()
        meta = manifest.get(f.name, {})
        stat = f.stat()
        results.append(
            CheckpointInfo(
                name=f.name,
                path=f,
                version=ver,
                size_mb=round(stat.st_size / (1024 * 1024), 1),
                modified=stat.st_mtime,
                is_latest=is_latest,
                train_loss=float(meta["train_loss"]) if "train_loss" in meta and meta["train_loss"] is not None else None,
                val_loss=float(meta["val_loss"]) if "val_loss" in meta and meta["val_loss"] is not None else None,
                step=int(meta["step"]) if "step" in meta and meta["step"] is not None else None,
                epoch=int(meta["epoch"]) if "epoch" in meta and meta["epoch"] is not None else None,
                saved_at=str(meta["saved_at"]) if "saved_at" in meta and meta["saved_at"] is not None else None,
            )
        )
    results.sort(key=lambda x: x.version, reverse=True)
    return results

__all__ = [
    "CheckpointInfo",
    "CHECKPOINTS_DIR",
    "DEFAULT_CHECKPOINT",
    "resolve_default_checkpoint",
    "get_latest_checkpoint",
    "get_next_checkpoint_path",
    "list_all_checkpoints",
]
