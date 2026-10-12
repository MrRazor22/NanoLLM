import os
from typing import Optional

def resolve_device(preferred: Optional[str] = None) -> str:
    """Resolve compute target device ('cuda', 'mps', 'cpu') without leaking substrate types."""
    if preferred and preferred.strip():
        return preferred.strip().lower()

    env = os.environ.get("PIQO_DEVICE")
    if env and env.strip():
        return env.strip().lower()

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"

__all__ = ["resolve_device"]
