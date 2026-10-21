from typing import Protocol, runtime_checkable
import torch
import torch.nn.functional as F

@runtime_checkable
class ILoss(Protocol):
    def compute(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...

class CalibratedLoss(ILoss):
    def __init__(self, brier_weight: float = 0.5):
        self.brier_weight = brier_weight

    def compute(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(scores, targets, reduction="mean")
        if self.brier_weight > 0:
            probabilities = F.softmax(scores, dim=1)
            one_hot_targets = F.one_hot(targets, num_classes=scores.size(1)).float()
            brier_loss = torch.mean(torch.sum((probabilities - one_hot_targets) ** 2, dim=1))
            return (1.0 - self.brier_weight) * cross_entropy + self.brier_weight * brier_loss
        return cross_entropy
