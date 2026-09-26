from typing import Any, List, Protocol
import torch
import torch.nn as nn
import torch.nn.functional as F

class ILossPolicy(Protocol):
    def compute(self, scores: torch.Tensor, batch_meta: Any, device: torch.device) -> torch.Tensor: ...

class CalibratedLoss(nn.Module, ILossPolicy):
    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def forward(
        self,
        scores: torch.Tensor,
        batch_meta: List[List[Any]],
        device: torch.device,
    ) -> torch.Tensor:
        losses = []
        for b_idx, sample_meta in enumerate(batch_meta):
            for q_type, positions, target in sample_meta:
                logits = scores[b_idx, positions]
                if q_type == "choice":
                    t = torch.tensor([target], dtype=torch.long, device=device)
                    losses.append(F.cross_entropy(logits.unsqueeze(0), t))
                elif q_type == "noul":
                    t = torch.tensor(target, dtype=torch.float, device=device)
                    prob = torch.sigmoid(logits[0])
                    losses.append(
                        (1.0 - self.brier_weight) * F.binary_cross_entropy_with_logits(logits[0], t)
                        + self.brier_weight * (prob - t) ** 2
                    )
                elif q_type == "score":
                    t = torch.tensor(target, dtype=torch.float, device=device)
                    losses.append(4.0 * (torch.sigmoid(logits[0]) - t) ** 2)
        return torch.stack(losses).sum() / max(1, len(losses)) if losses else torch.tensor(0.0, device=device)

    compute = forward
