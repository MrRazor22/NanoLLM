import json
from pathlib import Path
from typing import Optional, Sequence

from model.policies.weight_cache import DEFAULT_WEIGHTS_DIR
from model.weights import Weights

def list_weights(root: Optional[Path] = None) -> Sequence[Weights]:
    """Domain Extension: Discovers saved weights and their metadata from weights directory."""
    weights_dir = root or DEFAULT_WEIGHTS_DIR
    if not weights_dir.exists():
        return []

    results = []
    for w_file in weights_dir.glob("*.pt"):
        meta_file = weights_dir / f"{w_file.stem}.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        results.append(
            Weights(
                name=meta.get("name", w_file.name),
                size_mb=round(w_file.stat().st_size / (1024 * 1024), 1),
                epoch=meta.get("epoch", 1),
                train_loss=meta.get("train_loss"),
                val_loss=meta.get("val_loss"),
                saved_at=meta.get("saved_at"),
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
