from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

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

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.backbone is not None:
            hidden = self.backbone(input_ids=input_ids, attention_mask=mask).last_hidden_state
        else:
            hidden = self.tok_emb(input_ids)
        return self.head(hidden).squeeze(-1)
