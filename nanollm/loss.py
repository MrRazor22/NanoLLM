from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibratedLoss(nn.Module):
    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        types: List[str],
        opt_vectors: Optional[torch.Tensor] = None,
        opt_slices: Optional[List[List[int]]] = None
    ) -> torch.Tensor:
        b = targets.shape[0]
        total_loss = torch.tensor(0.0, device=targets.device)
        scale = outputs.get("scale", torch.tensor(1.0, device=targets.device))

        for idx, q_type in enumerate(types):
            q_target = targets[:, idx]
            if q_type == "choice":
                if opt_vectors is not None and opt_slices is not None:
                    choice_losses = []
                    for s_idx in range(b):
                        slices = opt_slices[s_idx]
                        if slices:
                            start, end = slices[0], slices[1]
                            opts = opt_vectors[start:end]
                            q_vec = outputs["q_choice"][s_idx, idx]
                            logits = scale * (q_vec @ opts.T)
                            t = q_target[s_idx].long().unsqueeze(0)
                            choice_losses.append(F.cross_entropy(logits.unsqueeze(0), t))
                    if choice_losses:
                        total_loss = total_loss + torch.stack(choice_losses).mean()
            elif q_type == "noul":
                noul_logits = outputs["noul"][:, idx]
                prob = torch.sigmoid(noul_logits)
                bce = F.binary_cross_entropy_with_logits(noul_logits, q_target)
                brier = F.mse_loss(prob, q_target)
                total_loss = total_loss + (1.0 - self.brier_weight) * bce + self.brier_weight * brier
            elif q_type == "score":
                score_logits = outputs["score"][:, idx]
                prob = torch.sigmoid(score_logits)
                total_loss = total_loss + F.mse_loss(prob, q_target)

        return total_loss / max(1, len(types))
