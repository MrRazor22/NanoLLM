from typing import Protocol, runtime_checkable
import torch
import torch.nn as nn
import torch.nn.functional as F

@runtime_checkable
class ILoss(Protocol):
    """Injected Opinion Sub-Primitive: Evaluates calibrated classification loss over candidate scores and targets."""
    def compute(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...

class CalibratedLoss(nn.Module, ILoss):
    """Brier-calibrated decision loss policy combining Cross-Entropy and Brier score."""

    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def compute(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(scores, targets, reduction="mean")

        if self.brier_weight > 0:
            probs = F.softmax(scores, dim=1)
            expected_probs = F.one_hot(targets, num_classes=scores.size(1)).float()
            brier_loss = torch.mean(torch.sum((probs - expected_probs) ** 2, dim=1))
            return (1.0 - self.brier_weight) * ce_loss + self.brier_weight * brier_loss

        return ce_loss
