from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50280
    hidden_dim: int = 768
    num_layers: int = 22
    num_heads: int = 12
    max_seq_len: int = 8192

class NanoModel(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.backbone = backbone
        hidden_dim = getattr(getattr(backbone, "config", None), "hidden_size", self.config.hidden_dim)
        if backbone is None:
            self.tok_emb = nn.Embedding(self.config.vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    @classmethod
    def from_backbone(cls, backbone_name: str = "answerdotai/ModernBERT-base", vocab_size: Optional[int] = None) -> "NanoModel":
        backbone = AutoModel.from_pretrained(backbone_name)
        cfg_model = getattr(backbone, "config", None)
        v_size = vocab_size or getattr(cfg_model, "vocab_size", 50280)
        h_dim = getattr(cfg_model, "hidden_size", 768)
        n_layers = getattr(cfg_model, "num_hidden_layers", 22)
        n_heads = getattr(cfg_model, "num_attention_heads", 12)
        config = ModelConfig(vocab_size=v_size, hidden_dim=h_dim, num_layers=n_layers, num_heads=n_heads)
        return cls(config, backbone=backbone)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.backbone is not None:
            hidden = self.backbone(input_ids=input_ids, attention_mask=mask).last_hidden_state
        else:
            hidden = self.tok_emb(input_ids)
        return self.head(hidden).squeeze(-1)

__all__ = ["ModelConfig", "NanoModel"]
