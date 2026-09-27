from typing import Any, List, Protocol, runtime_checkable
import torch
import torch.nn as nn
import torch.nn.functional as F

@runtime_checkable
class ILoss(Protocol):
    def compute(self, scores: torch.Tensor, batch_meta: Any, device: torch.device) -> torch.Tensor: ...

class CalibratedLoss(nn.Module, ILoss):
    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def forward(
        self,
        scores: torch.Tensor,
        batch_meta: List[List[Any]],
        device: torch.device,
    ) -> torch.Tensor:
        choice_logits, choice_targets = [], []
        bce_logits, bce_targets = [], []
        score_logits, score_targets = [], []

        for b_idx, sample_meta in enumerate(batch_meta):
            for q_type, positions, target in sample_meta:
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
            padded_logits = torch.full((len(choice_logits), max_c), -1e4, device=device)
            for i, l in enumerate(choice_logits):
                padded_logits[i, :l.size(0)] = l
            t = torch.as_tensor(choice_targets, dtype=torch.long, device=device)
            losses.append(F.cross_entropy(padded_logits, t, reduction="sum"))

        if bce_logits:
            l = torch.stack(bce_logits)
            t = torch.as_tensor(bce_targets, dtype=torch.float, device=device)
            prob = torch.sigmoid(l)
            bce_loss = (1.0 - self.brier_weight) * F.binary_cross_entropy_with_logits(l, t, reduction="sum") + self.brier_weight * torch.sum((prob - t) ** 2)
            losses.append(bce_loss)

        if score_logits:
            l = torch.stack(score_logits)
            t = torch.as_tensor(score_targets, dtype=torch.float, device=device)
            losses.append(torch.sum(4.0 * (torch.sigmoid(l) - t) ** 2))

        total_questions = len(choice_logits) + len(bce_logits) + len(score_logits)
        return torch.sum(torch.stack(losses)) / max(1, total_questions) if losses else torch.tensor(0.0, device=device)

    compute = forward
