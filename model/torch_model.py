from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, logging
from safetensors.torch import save_file, load_file

logging.set_verbosity_error()

from data.case import DecisionCase
from model.device import is_cuda, resolve_device
from model.layout import BatchTokens, ILayout, MaskedLayout
from model.weights import DEFAULT_WEIGHTS_PATH, WeightInfo, resolve_weights_path

@runtime_checkable
class IModel(Protocol):
    def forward(self, cases: Sequence[DecisionCase], train: bool = False) -> torch.Tensor: ...
    def save(self, path: Path = DEFAULT_WEIGHTS_PATH, info: Optional[WeightInfo] = None) -> None: ...
    def load(self, path: Path = DEFAULT_WEIGHTS_PATH) -> None: ...

class TorchModel(nn.Module, IModel):
    def __init__(
        self,
        backbone: str = "answerdotai/ModernBERT-base",
        head: Optional[nn.Module] = None,
        layout: Optional[ILayout] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device = resolve_device(device)
        self.backbone = AutoModel.from_pretrained(backbone).to(self.device)
        hidden_dim = self.backbone.config.hidden_size

        self.head = (head or nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )).to(self.device)

        self.layout = layout or MaskedLayout(
            tokenizer=tokenizer or AutoTokenizer.from_pretrained(backbone),
            max_sequence_tokens=int(self.backbone.config.max_position_embeddings),
        )

    def forward(self, cases: Sequence[DecisionCase], train: bool = False) -> torch.Tensor:
        if self.training != train:
            self.train(train)
        
        batch_tokens = self.layout.encode(cases).to(self.device)

        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=self.device.type, enabled=train and is_cuda(self.device)):
            embeddings = self.backbone(input_ids=batch_tokens.input_ids, attention_mask=batch_tokens.attention_mask).last_hidden_state
            batch_scores = [self.head(embeddings[i, idx, :]).squeeze(-1) for i, idx in enumerate(batch_tokens.candidate_idx)]
            return torch.stack(batch_scores)

    def save(self, path: Path = DEFAULT_WEIGHTS_PATH, info: Optional[WeightInfo] = None) -> None:
        out_file = resolve_weights_path(path)
        resolved_info = info or WeightInfo(name=out_file.stem, path=out_file)
        save_file(self.state_dict(), str(out_file), metadata=resolved_info.to_metadata())

    def load(self, path: Path = DEFAULT_WEIGHTS_PATH) -> None:
        in_file = resolve_weights_path(path)
        if not in_file.exists():
            raise FileNotFoundError(f"Model weights not found at: {in_file}")
        self.load_state_dict(load_file(str(in_file)))
