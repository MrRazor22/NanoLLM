import math
from typing import Protocol, runtime_checkable

@runtime_checkable
class ISchedule(Protocol):
    def get_lr(self, step: int, total_steps: int) -> float: ...

class CosineSchedule(ISchedule):
    def __init__(self, base_lr: float = 2e-5, warmup_ratio: float = 0.05, min_lr_ratio: float = 0.1):
        self.base_lr = base_lr
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio

    def get_lr(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return self.base_lr
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        if step < warmup_steps:
            return self.base_lr * (float(step) / float(warmup_steps))
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            return self.base_lr
        progress = min(max(float(step - warmup_steps) / float(decay_steps), 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay)
