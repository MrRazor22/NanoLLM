from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

# ==========================================
# Pure Passive Training DTOs
# ==========================================

@dataclass(frozen=True)
class TrainingBatch:
    input_ids: Sequence[Sequence[int]]
    mask: Sequence[Sequence[float]]
    meta: Sequence[Sequence[Tuple[str, Sequence[int], Union[int, float]]]]

    def __getitem__(self, key: str) -> Union[Sequence[Sequence[int]], Sequence[Sequence[float]], Sequence[Sequence[Tuple[str, Sequence[int], Union[int, float]]]]]:
        if key == "input_ids":
            return self.input_ids
        if key == "mask":
            return self.mask
        if key == "meta":
            return self.meta
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

__all__ = ["TrainingBatch", "StepStats"]
