from typing import Optional, Protocol, runtime_checkable
import torch
import torch.nn as nn
from nanollm.engine.questions import ScoreLayout, TokenLayout

@runtime_checkable
class IScorer(Protocol):
    """Injected Opinion Sub-Primitive: Evaluates token sequence slots into scalar scores."""

    def score(self, layout: TokenLayout) -> ScoreLayout:
        ...

class _ScoringHead(nn.Module):
    """Private neural head: attaches a 1D scalar linear head to the backbone."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        hidden_dim = getattr(getattr(backbone, "config", None), "hidden_size", 768)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=mask).last_hidden_state
        return self.head(hidden).squeeze(-1)

class TorchScorer(IScorer):
    """Concrete PyTorch neural scorer: encapsulates GPU tensors behind the pure DTO boundary."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def score(self, layout: TokenLayout) -> ScoreLayout:
        ids = torch.tensor(layout.input_ids, dtype=torch.long, device=self.device)
        mask = torch.tensor(layout.mask, dtype=torch.float, device=self.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                logits = self.model(ids, mask)
        return ScoreLayout(scores=logits.cpu().tolist())

__all__ = ["IScorer", "TorchScorer", "_ScoringHead"]
