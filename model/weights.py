from pathlib import Path
from typing import Optional, Sequence, Union
import torch.nn as nn
from model.policies.weight_cache import (
    DEFAULT_WEIGHTS_DIR, WeightMeta, Weights, DiskWeightCache,
)

def list_weights(root: Optional[Path] = None) -> Sequence[Weights]:
    return DiskWeightCache(root=root).list()

def has_weights(name: str = "piqo", root: Optional[Path] = None) -> bool:
    return any(w.name == name for w in list_weights(root))

def save_weights(
    model: nn.Module,
    target: Union[str, Path] = "piqo",
    meta: WeightMeta = WeightMeta(),
    root: Optional[Path] = None,
) -> Path:
    from model.torch_model import IModel
    resolved = Path(target)
    if isinstance(model, IModel) and hasattr(model, "save"):
        return model.save(resolved, meta=meta)
    from safetensors.torch import save
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    weights = Weights(payload=save(tensors), name=resolved.stem, meta=meta)
    return DiskWeightCache(root=root).put(resolved, weights)

def load_weights(
    model: nn.Module,
    source: Union[str, Path] = "piqo",
    root: Optional[Path] = None,
) -> nn.Module:
    from model.torch_model import IModel
    resolved = Path(source)
    if isinstance(model, IModel) and hasattr(model, "load"):
        model.load(resolved)
        return model
    from safetensors.torch import load
    weights = DiskWeightCache(root=root).get(resolved)
    if not weights:
        raise FileNotFoundError(f"Weights not found: {resolved}")
    model.load_state_dict(load(weights.payload))
    return model
