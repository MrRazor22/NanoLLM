import io
from pathlib import Path
import time
from typing import TYPE_CHECKING, Optional, Protocol, Sequence, Union, runtime_checkable
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, logging

logging.set_verbosity_error()

if TYPE_CHECKING:
    from data.case import DecisionCase

from inference.questions import DecisionState, Question
from model.model_extensions import resolve_device
from model.policies.calibrated_loss import CalibratedLoss, ILoss
from model.policies.head import IHead, MLPHead
from model.policies.layout_policy import ILayoutPolicy, MaskedLayoutPolicy
from model.policies.weight_cache import DiskWeightCache, IWeightCache
from model.token_layout import PredictionLayout
from model.weights import WeightMeta, Weights

@runtime_checkable
class IModel(Protocol):
    """Core Primitive: Neural computation boundary for inference scoring and loss computation."""
    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout: ...
    def step(self, cases: Sequence["DecisionCase"], backward: bool = True) -> float: ...
    def save(self, target: Union[str, Path] = "piqo", meta: Optional[WeightMeta] = None) -> Path: ...
    def load(self, source: Optional[Union[Weights, str, Path]] = None) -> None: ...

class TorchModel(nn.Module, IModel):
    """Core Primitive Implementation: ModernBERT backbone with modular decision head."""

    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-base",
        backbone: Optional[nn.Module] = None,
        head: Optional[IHead] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        weights: Optional[Weights] = None,
        loss_policy: Optional[ILoss] = None,
        weight_cache: Optional[IWeightCache] = None,
        layout_policy: Optional[ILayoutPolicy] = None,
        max_state_tokens: int = 384,
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
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(backbone_name)
        hidden_dim = getattr(self.backbone.config, "hidden_size", 768) if hasattr(self.backbone, "config") else 768

        self.head = (head or MLPHead(hidden_dim=hidden_dim)).to(self.device)
        self.weight_cache = weight_cache or DiskWeightCache()
        self.layout_policy = layout_policy or MaskedLayoutPolicy()
        self.loss_policy = loss_policy or CalibratedLoss()
        self.max_state_tokens = max_state_tokens
        self._tok_cache: dict[str, list[int]] = {}

        if weights is not None:
            self.load(weights)
        if compile and hasattr(torch, "compile"):
            self.backbone = torch.compile(self.backbone)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head.score(hidden)

    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout:
        layout = self.layout_policy.encode(self.tokenizer, state, questions, max_state_tokens=self.max_state_tokens, tok_cache=self._tok_cache)
        self.eval()
        ids = torch.as_tensor(layout.input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(layout.mask, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            scores = self(ids, mask).squeeze(0).cpu().tolist()
        return PredictionLayout(scores=scores, slots=layout.slots)

    def step(self, cases: Sequence["DecisionCase"], backward: bool = True) -> float:
        batch = self.layout_policy.collate(
            self.tokenizer, cases, max_state_tokens=self.max_state_tokens, shuffle_options=backward, tok_cache=self._tok_cache
        )
        self.train(backward)
        ids = batch.input_ids.to(self.device, non_blocking=True)
        mask = batch.mask.to(self.device, non_blocking=True)
        with torch.set_grad_enabled(backward), torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            preds = self(ids, mask)
            loss = self.loss_policy.compute(preds, batch.targets)
            if backward:
                loss.backward()
        return loss.item()

    def save(self, target: Union[str, Path] = "piqo", meta: Optional[WeightMeta] = None) -> Path:
        buf = io.BytesIO()
        torch.save(self.state_dict(), buf)
        metadata = meta or WeightMeta(saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        if not metadata.saved_at:
            metadata = WeightMeta(epoch=metadata.epoch, train_loss=metadata.train_loss, val_loss=metadata.val_loss, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        return self.weight_cache.put(target, Weights(payload=buf.getvalue(), name=Path(target).stem, meta=metadata))

    def load(self, source: Optional[Union[Weights, str, Path]] = None) -> None:
        target = "piqo" if source is None else source
        weights = target if isinstance(target, Weights) else self.weight_cache.get(target)
        if not weights:
            raise FileNotFoundError(f"Weights not found: {target}")
        if not weights.payload:
            return
        sd = torch.load(io.BytesIO(weights.payload), map_location=self.device)
        self.load_state_dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd, strict=False)
