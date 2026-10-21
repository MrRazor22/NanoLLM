import logging
from typing import Optional
import torch

logger = logging.getLogger("piqo.device")

CPU = "cpu"
CUDA = "cuda"

def configure_backend(device_name: str) -> None:
    if device_name.startswith(CUDA) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("Configured %s substrate optimizations (TF32=True, cuDNN benchmark=True)", device_name)

def resolve_device(device: Optional[str] = None) -> torch.device:
    if device and device.strip():
        target = torch.device(device.strip().lower())
        configure_backend(target.type)
        return target

    if torch.cuda.is_available():
        configure_backend(CUDA)
        return torch.device(CUDA)

    return torch.device(CPU)

def is_cuda(device: Optional[torch.device | str] = None) -> bool:
    if device is not None:
        target = device if isinstance(device, torch.device) else torch.device(device)
        return target.type == CUDA
    return resolve_device().type == CUDA

def seed_device(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_vram_gb() -> Optional[float]:
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / (1024**3)
    except (ImportError, AttributeError):
        pass
    return None

def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
