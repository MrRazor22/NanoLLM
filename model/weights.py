from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence
from safetensors import safe_open

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_WEIGHTS_PATH = DEFAULT_WEIGHTS_DIR / "piqo.safetensors"

def resolve_weights_path(path: Path | str = DEFAULT_WEIGHTS_PATH, ensure_parent: bool = False) -> Path:
    target = Path(path)
    file_path = target.with_suffix(".safetensors") if not target.suffix else target
    resolved = DEFAULT_WEIGHTS_DIR / file_path.name if file_path.parent == Path(".") else file_path
    if ensure_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

@dataclass(frozen=True, slots=True)
class WeightInfo:
    name: str = "piqo"
    path: Optional[Path] = None
    size_mb: float = 0.0
    saved_at: Optional[str] = None

    def to_metadata(self) -> dict[str, str]:
        return {
            k: str(v)
            for k, v in asdict(self).items()
            if v is not None and k not in ("path", "size_mb")
        }

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
                name=meta.get("name", f.stem),
                path=f,
                size_mb=round(f.stat().st_size / (1024 * 1024), 1),
                saved_at=meta.get("saved_at"),
            )
        )
    return sorted(results, key=lambda x: x.saved_at or "", reverse=True)
