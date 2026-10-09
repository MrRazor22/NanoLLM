from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

# ==========================================
# Pure Passive Training DTOs
# ==========================================

@dataclass(frozen=True)
class TargetSlot:
    target_type: str
    indices: Sequence[int]
    value: Union[int, float]

@dataclass(frozen=True)
class SampleTargets:
    slots: Sequence[TargetSlot]

@dataclass(frozen=True)
class TrainingBatch:
    input_ids: Sequence[Sequence[int]]
    mask: Sequence[Sequence[float]]
    targets: Sequence[SampleTargets]

    def __getitem__(self, key: str) -> Union[Sequence[Sequence[int]], Sequence[Sequence[float]], Sequence[SampleTargets]]:
        if key == "input_ids":
            return self.input_ids
        if key == "mask":
            return self.mask
        if key in ("targets", "meta"):
            return self.targets
        raise KeyError(key)

@dataclass(frozen=True)
class StepStats:
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

__all__ = ["TargetSlot", "SampleTargets", "TrainingBatch", "StepStats"]
