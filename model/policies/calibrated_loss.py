from typing import Protocol, Sequence, runtime_checkable
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.token_layout import SampleTargets

@runtime_checkable
class ILoss(Protocol):
    """Injected Opinion Sub-Primitive: Evaluates multi-task loss over predictions and targets."""
    def compute(self, predictions: torch.Tensor, targets: Sequence[SampleTargets]) -> torch.Tensor: ...

class CalibratedLoss(nn.Module, ILoss):
    """Brier-calibrated multi-task loss policy: vectorizes cross-entropy, Brier, and MSE losses."""

    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def compute(self, predictions: torch.Tensor, targets: Sequence[SampleTargets]) -> torch.Tensor:
        dev = predictions.device
        logits_list, tgts_list = [], []

        for b, sample in enumerate(targets):
            for item in sample.slots:
                logits_list.append(predictions[b, item.indices])
                tgts_list.append(int(item.value))

        if not logits_list:
            return torch.tensor(0.0, device=dev)

        max_c = max(l.size(0) for l in logits_list)
        pad = torch.full((len(logits_list), max_c), -1e4, device=dev)
        for i, l in enumerate(logits_list):
            pad[i, :l.size(0)] = l
        tgt_tensor = torch.as_tensor(tgts_list, dtype=torch.long, device=dev)
        ce_loss = F.cross_entropy(pad, tgt_tensor, reduction="mean")

        if self.brier_weight > 0:
            probs = F.softmax(pad, dim=-1)
            one_hot = F.one_hot(tgt_tensor, num_classes=max_c).float()
            brier_loss = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))
            return (1.0 - self.brier_weight) * ce_loss + self.brier_weight * brier_loss

        return ce_loss
