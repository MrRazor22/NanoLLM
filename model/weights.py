from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from safetensors import safe_open

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_WEIGHTS_PATH = DEFAULT_WEIGHTS_DIR / "piqo.safetensors"

def resolve_weights_path(path: Path | str = DEFAULT_WEIGHTS_PATH) -> Path:
    target = Path(path)
    file_path = target.with_suffix(".safetensors") if not target.suffix else target
    return DEFAULT_WEIGHTS_DIR / file_path.name if file_path.parent == Path(".") else file_path

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
