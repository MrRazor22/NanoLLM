from typing import Protocol, Sequence, runtime_checkable
import torch
import torch.nn as nn

@runtime_checkable
class IHead(Protocol):
    """Opinion Sub-Primitive: Projects token representations to scalar decision scores."""
    def score(self, representations: Sequence) -> Sequence: ...

class MLPHead(nn.Sequential, IHead):
    """Default decision head: LayerNorm + Dropout + Linear + GELU + Linear."""

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def score(self, representations: Sequence) -> Sequence:
        if not isinstance(representations, torch.Tensor):
            representations = torch.as_tensor(representations, dtype=torch.float)
        return self(representations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).squeeze(-1)
