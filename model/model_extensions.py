import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

if TYPE_CHECKING:
    from model.policies.weight_cache import IWeightCache
    from model.torch_model import IModel

from model.policies.weight_cache import DEFAULT_WEIGHTS_DIR
from model.weights import WeightMeta, Weights

def list_weights(root: Optional[Path] = None) -> Sequence[Weights]:
    """Domain Extension: Discovers saved weights and their metadata from weights directory."""
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
    return sorted(results, key=lambda x: x.saved_at or "", reverse=True)

def resolve_device(preferred: Optional[str] = None) -> str:
    """Domain Extension: Resolves optimal compute target device name without substrate leakage."""
    import os
    import torch

    if preferred and preferred.strip():
        return preferred.strip().lower()

    env = os.environ.get("PIQO_DEVICE")
    if env and env.strip():
        return env.strip().lower()

    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except (ImportError, AttributeError):
        pass

    return "cpu"

def save_weights(model: "IModel", target: Union[str, Path] = "piqo", stats: Optional[Any] = None, cache: Optional["IWeightCache"] = None) -> Path:
    """Domain Extension: Persists model weights and optional metadata through weight cache policy."""
    import time
    t_loss = getattr(stats, "train_loss", getattr(stats, "loss", None)) if stats else None
    v_loss = getattr(stats, "val_loss", None) if stats else None
    meta = WeightMeta(
        epoch=getattr(stats, "epoch", 1) if stats else 1,
        train_loss=round(t_loss, 6) if t_loss is not None else None,
        val_loss=round(v_loss, 6) if v_loss is not None else None,
        saved_at=getattr(stats, "saved_at", time.strftime("%Y-%m-%d %H:%M:%S")),
    )
    return model.save(target, meta=meta)

def load_weights(model: "IModel", source: Union[Weights, str, Path] = "piqo", cache: Optional["IWeightCache"] = None) -> None:
    """Domain Extension: Rehydrates model weights from Weights DTO or weight cache identifier/path."""
    from model.policies.weight_cache import DiskWeightCache
    if isinstance(source, Weights):
        model.load(source)
        return
    c = cache or DiskWeightCache()
    w = c.get(source)
    if not w:
        raise FileNotFoundError(f"Weights not found: {source}")
    model.load(w)
