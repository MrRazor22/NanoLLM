from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50280
    hidden_dim: int = 768
    num_layers: int = 22
    num_heads: int = 12
    max_seq_len: int = 8192
    proj_dim: int = 256

class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)
        return self.proj(attn.permute(0, 2, 1, 3).reshape(b, s, -1))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))

class NanoModel(nn.Module):
    def __init__(self, config: ModelConfig, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.backbone = backbone
        hidden_dim = backbone.config.hidden_size if backbone is not None else config.hidden_dim

        if backbone is None:
            self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
            self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)
            self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
            self.ln_f = nn.LayerNorm(config.hidden_dim)

        self.proj_choice = nn.Linear(hidden_dim, config.proj_dim)
        self.proj_option = nn.Linear(hidden_dim, config.proj_dim)
        self.noul_head = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def _encode(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.backbone is not None:
            return self.backbone(input_ids=input_ids, attention_mask=mask).last_hidden_state
        b, s = input_ids.shape
        pos = torch.arange(0, s, device=input_ids.device)
        attn_mask = mask[:, None, None, :].bool() if mask is not None and mask.dim() == 2 else mask
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.ln_f(x)

    def encode_options(self, opt_ids: torch.Tensor, opt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._encode(opt_ids, opt_mask)
        if opt_mask is not None:
            mask_exp = opt_mask.unsqueeze(-1)
            pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        return F.normalize(self.proj_option(pooled), dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        question_positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        b, s = input_ids.shape
        x = self._encode(input_ids, mask)
        b_idx = torch.arange(b, device=input_ids.device).unsqueeze(1)
        q_features = x[b_idx, question_positions]
        q_choice = F.normalize(self.proj_choice(q_features), dim=-1)
        return {
            "q_choice": q_choice,
            "noul": self.noul_head(q_features).squeeze(-1),
            "score": self.score_head(q_features).squeeze(-1),
            "scale": self.logit_scale.exp().clamp(max=100.0)
        }
