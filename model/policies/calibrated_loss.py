from typing import Protocol, Sequence, runtime_checkable
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.token_layout import SampleTargets

@runtime_checkable
class ILoss(Protocol):
    """Injected Opinion Sub-Primitive: Evaluates multi-task loss over predictions and targets."""
    def compute(self, predictions: Sequence, targets: Sequence[SampleTargets]) -> float: ...

class CalibratedLoss(nn.Module, ILoss):
    """Brier-calibrated multi-task loss policy: vectorizes cross-entropy, Brier, and MSE losses."""

    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def compute(self, predictions, targets: Sequence[SampleTargets]):
        dev = predictions.device
        c_logits, c_tgts = [], []
        b_idx, b_pos, b_tgts = [], [], []
        s_idx, s_pos, s_tgts = [], [], []

        for b, sample in enumerate(targets):
            for item in sample.slots:
                t, pos, val = item.target_type, item.indices, item.value
                if t == "choice":
                    c_logits.append(predictions[b, pos])
                    c_tgts.append(int(val))
                elif t == "noul":
                    b_idx.append(b); b_pos.append(pos[0]); b_tgts.append(float(val))
                elif t == "score":
                    s_idx.append(b); s_pos.append(pos[0]); s_tgts.append(float(val))

        losses, counts = [], []
        if c_logits:
            max_c = max(l.size(0) for l in c_logits)
            pad = torch.full((len(c_logits), max_c), -1e4, device=dev)
            for i, l in enumerate(c_logits):
                pad[i, :l.size(0)] = l
            tgt_tensor = torch.as_tensor(c_tgts, dtype=torch.long, device=dev)
            ce_loss = F.cross_entropy(pad, tgt_tensor, reduction="mean")
            if self.brier_weight > 0:
                probs = F.softmax(pad, dim=-1)
                one_hot = F.one_hot(tgt_tensor, num_classes=max_c).float()
                brier_loss = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))
                losses.append((1.0 - self.brier_weight) * ce_loss + self.brier_weight * brier_loss)
            else:
                losses.append(ce_loss)
            counts.append(len(c_logits))
        if b_idx:
            pred = predictions[torch.as_tensor(b_idx, device=dev), torch.as_tensor(b_pos, device=dev)]
            tgt = torch.as_tensor(b_tgts, dtype=torch.float, device=dev)
            prob = torch.sigmoid(pred)
            bce_loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="mean")
            brier_bce = torch.mean((prob - tgt) ** 2)
            losses.append((1.0 - self.brier_weight) * bce_loss + self.brier_weight * brier_bce)
            counts.append(len(b_idx))
        if s_idx:
            pred = predictions[torch.as_tensor(s_idx, device=dev), torch.as_tensor(s_pos, device=dev)]
            tgt = torch.as_tensor(s_tgts, dtype=torch.float, device=dev)
            losses.append(torch.mean(4.0 * (torch.sigmoid(pred) - tgt) ** 2))
            counts.append(len(s_idx))

        if not losses:
            return torch.tensor(0.0, device=dev)
        weights = torch.as_tensor(counts, dtype=torch.float, device=dev)
        return torch.sum(torch.stack(losses) * weights) / torch.sum(weights)

    forward = compute
