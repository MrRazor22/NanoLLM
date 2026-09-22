from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibratedLoss(nn.Module):
    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        types: List[str]
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=logits.device)
        for idx, q_type in enumerate(types):
            q_logits = logits[:, idx]
            q_target = targets[:, idx]
            if q_type == "choice":
                total_loss = total_loss + F.cross_entropy(q_logits, q_target.long())
            elif q_type == "noul":
                prob = torch.sigmoid(q_logits[:, 0])
                bce = F.binary_cross_entropy_with_logits(q_logits[:, 0], q_target)
                brier = F.mse_loss(prob, q_target)
                total_loss = total_loss + (1.0 - self.brier_weight) * bce + self.brier_weight * brier
            elif q_type == "score":
                prob = torch.sigmoid(q_logits[:, 0])
                total_loss = total_loss + F.mse_loss(prob, q_target)
        return total_loss / max(1, len(types))
