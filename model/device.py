import os
from typing import Optional
import torch

def resolve_device(preferred: Optional[str] = None) -> str:
    """Domain Operation: Resolves optimal compute target device name without substrate leakage."""
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

def resolve_vram_gb() -> Optional[float]:
    """Domain Operation: Inspects GPU reserved memory in GB without substrate leakage."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / (1024**3)
    except (ImportError, AttributeError):
        pass
    return None

def synchronize_device() -> None:
    """Domain Operation: Blocks until compute substrate finishes asynchronous execution."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
