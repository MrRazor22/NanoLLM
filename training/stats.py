from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class StepStats:
    """Immutable domain DTO for step-level optimization metrics."""
    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    loss: float
    elapsed_ms: float
    lr: float = 0.0
    val_loss: Optional[float] = None
    batch: int = 0
    total_batches: int = 0

__all__ = ["StepStats"]
