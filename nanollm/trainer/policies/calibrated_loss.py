from typing import Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanollm.trainer.batches import SampleTargets, TargetSlot

@runtime_checkable
class ILoss(Protocol):
    """Injected Opinion Sub-Primitive: Evaluates scalar loss over neural logits and batch targets."""

    def compute(
        self,
        scores: Sequence[Sequence[float]],
        batch_targets: Sequence[SampleTargets],
    ) -> float:
        ...

class CalibratedLoss(nn.Module, ILoss):
    """Brier-calibrated multi-task loss policy."""

    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def compute(
        self,
        scores: Sequence[Sequence[float]],
        batch_targets: Sequence[SampleTargets],
    ) -> float:
        t_scores = torch.as_tensor(scores, dtype=torch.float)
        return float(self.loss_tensor(t_scores, batch_targets).item())

    def loss_tensor(
        self,
        scores: torch.Tensor,
        batch_meta: Union[Sequence[SampleTargets], Sequence[Sequence[Union[TargetSlot, Tuple[str, Sequence[int], Union[int, float]]]]]],
    ) -> torch.Tensor:
        device = scores.device
        choice_logits, choice_targets = [], []
        bce_logits, bce_targets = [], []
        score_logits, score_targets = [], []

        for b_idx, sample in enumerate(batch_meta):
            items = sample.slots if isinstance(sample, SampleTargets) else sample
            for item in items:
                q_type = item.target_type if isinstance(item, TargetSlot) else item[0]
                positions = item.indices if isinstance(item, TargetSlot) else item[1]
                target = item.value if isinstance(item, TargetSlot) else item[2]
                if q_type == "choice":
                    choice_logits.append(scores[b_idx, positions])
                    choice_targets.append(int(target))
                elif q_type == "noul":
                    bce_logits.append(scores[b_idx, positions[0]])
                    bce_targets.append(float(target))
                elif q_type == "score":
                    score_logits.append(scores[b_idx, positions[0]])
                    score_targets.append(float(target))

        losses = []
        if choice_logits:
            max_c = max(l.size(0) for l in choice_logits)
            padded = torch.full((len(choice_logits), max_c), -1e4, device=device)
            for i, l in enumerate(choice_logits):
                padded[i, :l.size(0)] = l
            t = torch.as_tensor(choice_targets, dtype=torch.long, device=device)
            losses.append(F.cross_entropy(padded, t, reduction="sum"))

        if bce_logits:
            l = torch.stack(bce_logits)
            t = torch.as_tensor(bce_targets, dtype=torch.float, device=device)
            prob = torch.sigmoid(l)
            bce = (1.0 - self.brier_weight) * F.binary_cross_entropy_with_logits(l, t, reduction="sum") + self.brier_weight * torch.sum((prob - t) ** 2)
            losses.append(bce)

        if score_logits:
            l = torch.stack(score_logits)
            t = torch.as_tensor(score_targets, dtype=torch.float, device=device)
            losses.append(torch.sum(4.0 * (torch.sigmoid(l) - t) ** 2))

        total_q = len(choice_logits) + len(bce_logits) + len(score_logits)
        return torch.sum(torch.stack(losses)) / max(1, total_q) if losses else torch.tensor(0.0, device=device)

    def forward(
        self,
        scores: torch.Tensor,
        batch_meta: Sequence[Sequence[Union[TargetSlot, Tuple[str, Sequence[int], Union[int, float]]]]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.loss_tensor(scores, batch_meta)

__all__ = ["ILoss", "CalibratedLoss"]
