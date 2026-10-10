from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable
import torch
import torch.nn as nn
from transformers import AutoModel
from inference.questions import ScoreLayout, TokenLayout

@runtime_checkable
class IScorer(Protocol):
    """Injected Opinion Sub-Primitive: Evaluates token sequence slots into scalar scores."""
    def score(self, layout: TokenLayout) -> ScoreLayout: ...

class _ScoringHead(nn.Module):
    """Private neural head: attaches a 1D scalar linear head to the backbone."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        hidden_dim = getattr(backbone.config, "hidden_size", 768) if hasattr(backbone, "config") else 768
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=mask).last_hidden_state
        return self.head(hidden).squeeze(-1)

    @classmethod
    def load(
        cls,
        backbone_name: str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        compile: bool = False,
    ) -> nn.Module:
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = AutoModel.from_pretrained(backbone_name)
        model = cls(backbone).to(dev)
        if checkpoint_path:
            ckpt = torch.load(str(checkpoint_path), map_location=dev)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
        if compile and hasattr(torch, "compile"):
            model = torch.compile(model)
        return model

class TorchScorer(IScorer):
    """Concrete PyTorch neural scorer: encapsulates model weights and GPU execution."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

    @classmethod
    def from_backbone(
        cls,
        backbone_name: str = "answerdotai/ModernBERT-base",
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        compile: bool = False,
    ) -> "TorchScorer":
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _ScoringHead.load(backbone_name, checkpoint_path, dev, compile)
        return cls(model, dev)

    def score(self, layout: TokenLayout) -> ScoreLayout:
        ids = torch.as_tensor(layout.input_ids, dtype=torch.long, device=self.device)
        mask = torch.as_tensor(layout.mask, dtype=torch.float, device=self.device)
        if ids.ndim == 1:
            ids, mask = ids.unsqueeze(0), mask.unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self.model(ids, mask)
        raw = logits.squeeze(0).cpu().tolist() if logits.ndim > 1 else logits.cpu().tolist()
        return ScoreLayout(scores=raw)

__all__ = ["IScorer", "TorchScorer", "_ScoringHead"]
