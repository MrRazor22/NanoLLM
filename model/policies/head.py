from typing import Protocol, runtime_checkable
import torch
import torch.nn as nn

@runtime_checkable
class IHead(Protocol):
    """Opinion Sub-Primitive: Projects hidden token representations to scalar decision scores."""
    def score(self, representations: torch.Tensor) -> torch.Tensor: ...

class MLPHead(nn.Module, IHead):
    """Default decision head: opinion sub-primitive composing LayerNorm + Dropout + Linear + GELU + Linear."""

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def score(self, representations: torch.Tensor) -> torch.Tensor:
        return self._net(representations).squeeze(-1)

