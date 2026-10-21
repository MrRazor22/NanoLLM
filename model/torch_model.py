from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, logging
from safetensors.torch import save, load

logging.set_verbosity_error()

from data.case import DecisionCase
from model.device import resolve_device
from model.policies.layout import ILayout, MaskedLayout
from model.policies.weight_cache import (
    DEFAULT_WEIGHTS_PATH,
    IWeightCache,
    DiskWeightCache,
    WeightMeta,
    Weights,
)

@runtime_checkable
class IModel(Protocol):
    def forward(self, cases: Sequence[DecisionCase], train: bool = False) -> torch.Tensor: ...
    def save(self, path: Path = DEFAULT_WEIGHTS_PATH, meta: WeightMeta = WeightMeta()) -> Path: ...
    def load(self, path: Path = DEFAULT_WEIGHTS_PATH) -> "IModel": ...

class TorchModel(nn.Module, IModel):
    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-base",
        backbone: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        layout: Optional[ILayout] = None,
        cache: Optional[IWeightCache] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_state_tokens: int = 2048,
        device: Optional[Union[str, torch.device]] = None,
        compile: bool = False,
    ):
        super().__init__()
        target_dev = device if isinstance(device, torch.device) else resolve_device(device)
        self.device = torch.device(target_dev) if isinstance(target_dev, str) else target_dev
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model_kwargs = {"attn_implementation": "sdpa"} if self.device.type == "cuda" else {}
        self.backbone = (backbone or AutoModel.from_pretrained(backbone_name, **model_kwargs)).to(self.device)
        hidden_dim = getattr(self.backbone.config, "hidden_size", 768) if hasattr(self.backbone, "config") else 768

        self.head = (head or nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )).to(self.device)

        self.layout = layout or MaskedLayout(
            tokenizer=tokenizer or AutoTokenizer.from_pretrained(backbone_name),
            max_state_tokens=max_state_tokens,
        )
        self.cache = cache or DiskWeightCache()

        if compile and hasattr(torch, "compile"):
            self.backbone = torch.compile(self.backbone)

    def forward(self, cases: Sequence[DecisionCase], train: bool = False) -> torch.Tensor:
        layouts = self.layout.layout(cases)
        pad_id = layouts[0].pad_id if layouts else 0

        input_ids = pad_sequence([torch.as_tensor(l.tokens, dtype=torch.long) for l in layouts], batch_first=True, padding_value=pad_id).to(self.device)
        attention_mask = (input_ids != pad_id).long()

        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=self.device.type, enabled=train and self.device.type == "cuda"):
            embeddings = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            batch_scores = []
            for i, l in enumerate(layouts):
                slots = torch.as_tensor(l.candidate_indices, dtype=torch.long, device=self.device)
                slot_embeddings = embeddings[i, slots, :]
                scores = self.head(slot_embeddings).squeeze(-1)
                batch_scores.append(scores)
            return torch.stack(batch_scores)

    def save(self, path: Path = DEFAULT_WEIGHTS_PATH, meta: WeightMeta = WeightMeta()) -> Path:
        resolved_path = Path(path)
        tensors = {k: v.contiguous() for k, v in self.state_dict().items()}
        weights = Weights(payload=save(tensors), name=resolved_path.stem, meta=meta)
        return self.cache.put(resolved_path, weights)

    def load(self, path: Path = DEFAULT_WEIGHTS_PATH) -> "IModel":
        resolved_path = Path(path)
        weights = self.cache.get(resolved_path)
        if not weights:
            raise FileNotFoundError(f"Model weights not found for: {resolved_path}")
        self.load_state_dict(load(weights.payload))
        return self
