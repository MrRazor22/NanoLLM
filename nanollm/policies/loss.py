from typing import Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibratedLoss(nn.Module):
    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def forward(
        self,
        scores: torch.Tensor,
        batch_meta: List[List[Any]],
        device: torch.device,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for b_idx, sample_meta in enumerate(batch_meta):
            for q_type, positions, target in sample_meta:
                logits = scores[b_idx, positions]
                if q_type == "choice":
                    t = torch.tensor([target], dtype=torch.long, device=device)
                    total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), t)
                elif q_type == "noul":
                    t = torch.tensor(target, dtype=torch.float, device=device)
                    prob = torch.sigmoid(logits[0])
                    bce = F.binary_cross_entropy_with_logits(logits[0], t)
                    brier = (prob - t) ** 2
                    total_loss = total_loss + (1.0 - self.brier_weight) * bce + self.brier_weight * brier
                elif q_type == "score":
                    t = torch.tensor(target, dtype=torch.float, device=device)
                    total_loss = total_loss + (torch.sigmoid(logits[0]) - t) ** 2
                count += 1

        return total_loss / max(1, count)
