from typing import Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.batches import SampleTargets, TargetSlot

@runtime_checkable
class ILoss(Protocol):
    """Injected Opinion Sub-Primitive: Evaluates scalar loss over neural logits and batch targets."""
    def compute(self, scores: Sequence[Sequence[float]], batch_targets: Sequence[SampleTargets]) -> float: ...

class CalibratedLoss(nn.Module, ILoss):
    """Brier-calibrated multi-task loss policy."""
    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def compute(self, scores: Sequence[Sequence[float]], batch_targets: Sequence[SampleTargets]) -> float:
        return float(self.loss_tensor(torch.as_tensor(scores, dtype=torch.float), batch_targets).item())

    def loss_tensor(self, scores: torch.Tensor, batch_meta: Sequence[Sequence[TargetSlot]]) -> torch.Tensor:
        dev = scores.device
        c_logits, c_tgts, b_logits, b_tgts, s_logits, s_tgts = [], [], [], [], [], []
        for b_idx, sample in enumerate(batch_meta):
            items = sample.slots if hasattr(sample, "slots") else sample
            for item in items:
                q_type, pos, val = item.target_type, item.indices, item.value
                if q_type == "choice":
                    c_logits.append(scores[b_idx, pos])
                    c_tgts.append(int(val))
                elif q_type == "noul":
                    b_logits.append(scores[b_idx, pos[0]])
                    b_tgts.append(float(val))
                elif q_type == "score":
                    s_logits.append(scores[b_idx, pos[0]])
                    s_tgts.append(float(val))

        losses = []
        if c_logits:
            max_c = max(l.size(0) for l in c_logits)
            pad = torch.full((len(c_logits), max_c), -1e4, device=dev)
            for i, l in enumerate(c_logits): pad[i, :l.size(0)] = l
            losses.append(F.cross_entropy(pad, torch.as_tensor(c_tgts, dtype=torch.long, device=dev), reduction="sum"))
        if b_logits:
            l, t = torch.stack(b_logits), torch.as_tensor(b_tgts, dtype=torch.float, device=dev)
            prob = torch.sigmoid(l)
            losses.append((1.0 - self.brier_weight) * F.binary_cross_entropy_with_logits(l, t, reduction="sum") + self.brier_weight * torch.sum((prob - t) ** 2))
        if s_logits:
            l, t = torch.stack(s_logits), torch.as_tensor(s_tgts, dtype=torch.float, device=dev)
            losses.append(torch.sum(4.0 * (torch.sigmoid(l) - t) ** 2))

        total_q = len(c_logits) + len(b_logits) + len(s_logits)
        return torch.sum(torch.stack(losses)) / max(1, total_q) if losses else torch.tensor(0.0, device=dev)

    def forward(self, scores: torch.Tensor, batch_meta: Sequence[Sequence[TargetSlot]], device: Optional[torch.device] = None) -> torch.Tensor:
        return self.loss_tensor(scores, batch_meta)

__all__ = ["ILoss", "CalibratedLoss"]
