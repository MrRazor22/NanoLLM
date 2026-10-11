from dataclasses import dataclass
from typing import Sequence, Union

@dataclass(frozen=True)
class TokenLayout:
    """Explicit domain DTO for tokenized input sequence and question slot positions."""
    input_ids: Sequence[int]
    mask: Sequence[float]
    slots: Sequence[Sequence[int]]

@dataclass(frozen=True)
class ScoreLayout:
    """Explicit domain DTO for model output score distribution."""
    scores: Sequence[float]

@dataclass(frozen=True)
class TargetSlot:
    """Explicit domain DTO for a question's masked slot positions and target value."""
    target_type: str
    indices: Sequence[int]
    value: Union[int, float]

@dataclass(frozen=True)
class SampleTargets:
    """Explicit domain DTO for all target slots in a single sample."""
    slots: Sequence[TargetSlot]

@dataclass(frozen=True)
class Batch:
    """Explicit domain DTO for a batch of encoded tokens and targets."""
    input_ids: Sequence[Sequence[int]]
    mask: Sequence[Sequence[float]]
    targets: Sequence[SampleTargets]

__all__ = [
    "TokenLayout",
    "ScoreLayout",
    "TargetSlot",
    "SampleTargets",
    "Batch",
]
