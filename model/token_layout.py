from dataclasses import dataclass
from typing import Sequence
import torch

@dataclass(frozen=True, slots=True)
class TokenSlot:
    """Explicit domain DTO for a question's candidate token slot positions."""
    indices: Sequence[int]

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, idx: int) -> int:
        return self.indices[idx]

    def __len__(self) -> int:
        return len(self.indices)

@dataclass(frozen=True, slots=True)
class TokenLayout:
    """Explicit domain DTO for tokenized input sequence and question slot positions."""
    input_ids: Sequence[int]
    mask: Sequence[float]
    slots: Sequence[TokenSlot]

@dataclass(frozen=True, slots=True)
class PredictionLayout:
    """Explicit domain DTO for model output scores and corresponding question slot positions."""
    scores: Sequence[float]
    slots: Sequence[TokenSlot]

@dataclass(frozen=True, slots=True)
class TargetSlot:
    """Explicit domain DTO for a question's masked slot positions and target value."""
    target_type: str
    indices: Sequence[int]
    value: int | float

@dataclass(frozen=True, slots=True)
class SampleTargets:
    """Explicit domain DTO for all target slots in a single sample."""
    slots: Sequence[TargetSlot]

@dataclass(frozen=True, slots=True)
class Batch:
    """Explicit domain DTO for a batch of encoded tokens and targets."""
    input_ids: torch.Tensor | Sequence[Sequence[int]]
    mask: torch.Tensor | Sequence[Sequence[float]]
    targets: Sequence[SampleTargets]
