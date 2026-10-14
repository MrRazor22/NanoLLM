import math
from typing import Protocol, runtime_checkable

@runtime_checkable
class ISchedule(Protocol):
    """Policy Contract: determines parameter learning rate per step given total step budget."""
    def get_lr(self, step: int, total_steps: int, base_lr: float) -> float: ...

class CosineSchedule(ISchedule):
    """Linear warmup followed by cosine annealing decay."""

    def __init__(self, warmup_ratio: float = 0.05, min_lr_ratio: float = 0.1):
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio

    def get_lr(self, step: int, total_steps: int, base_lr: float) -> float:
        if total_steps <= 0:
            return base_lr
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        if step < warmup_steps:
            return base_lr * (float(step) / float(warmup_steps))
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            return base_lr
        progress = min(max(float(step - warmup_steps) / float(decay_steps), 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay)

class ConstantSchedule(ISchedule):
    """Preserves base learning rate across all steps unconditionally."""

    def get_lr(self, step: int, total_steps: int, base_lr: float) -> float:
        return base_lr
