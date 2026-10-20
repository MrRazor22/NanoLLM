from dataclasses import dataclass
from typing import Optional, Sequence
import torch

@dataclass(frozen=True, slots=True)
class Batch:
    """Unified domain DTO for encoded token sequences, attention masks, and candidate positions."""
    input_ids: torch.Tensor
    mask: torch.Tensor
    candidate_indices: Sequence[Sequence[int]]
    targets: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            mask=self.mask.to(device, non_blocking=True),
            candidate_indices=self.candidate_indices,
            targets=self.targets.to(device, non_blocking=True) if self.targets is not None else None,
        )
