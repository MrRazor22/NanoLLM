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
from model.policies.layout import ILayout, MaskedLayout
from model.policies.optimizer import AdamWOptimizer, IOptimizerPolicy
from model.policies.weight_cache import DiskWeightCache, IWeightCache
from model.token_layout import PredictionLayout
from model.weights import WeightMeta, Weights

@runtime_checkable
class IModel(Protocol):
    """Core Primitive: Neural computation boundary for inference scoring and loss computation."""
    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout: ...
    def step(self, cases: Sequence["DecisionCase"], backward: bool = True, lr: Optional[float] = None) -> float: ...
    def save(self, target: Union[str, Path] = "piqo", meta: Optional[WeightMeta] = None) -> Path: ...
    def load(self, source: Optional[Union[Weights, str, Path]] = None) -> None: ...

class TorchModel(IModel):
    """Core Primitive Implementation: ModernBERT backbone with modular decision head, layout, and optimizer."""

    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-base",
        backbone: Optional[nn.Module] = None,
        head: Optional[IHead] = None,
        layout: Optional[ILayout] = None,
        loss_policy: Optional[ILoss] = None,
        optimizer_policy: Optional[IOptimizerPolicy] = None,
        weight_cache: Optional[IWeightCache] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        weights: Optional[Weights] = None,
        lr: float = 2e-5,
        accum_steps: int = 1,
        max_state_tokens: int = 384,
        device: Optional[Union[str, torch.device]] = None,
        compile: bool = False,
    ):
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
        self.layout = layout or MaskedLayout(tokenizer=self.tokenizer, max_state_tokens=max_state_tokens)
        self.loss_policy = loss_policy or CalibratedLoss()
        self.weight_cache = weight_cache or DiskWeightCache()

        head_params = list(self.head.parameters()) if hasattr(self.head, "parameters") else []
        params = list(self.backbone.parameters()) + head_params
        self.optimizer_policy = optimizer_policy or AdamWOptimizer(params, lr=lr, accum_steps=accum_steps, device=self.device)

        if weights is not None:
            self.load(weights)
        if compile and hasattr(torch, "compile"):
            self.backbone = torch.compile(self.backbone)

    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout:
        layout = self.layout.encode(state, questions)
        self.backbone.eval()
        if hasattr(self.head, "eval"):
            self.head.eval()
        ids = torch.as_tensor(layout.input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(layout.mask, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            scores = self.head.score(hidden).squeeze(0).cpu().tolist()
        return PredictionLayout(scores=scores, slots=layout.slots)

    def step(self, cases: Sequence["DecisionCase"], backward: bool = True, lr: Optional[float] = None) -> float:
        batch = self.layout.collate(cases, shuffle=backward)
        self.backbone.train(backward)
        if hasattr(self.head, "train"):
            self.head.train(backward)
        ids = batch.input_ids.to(self.device, non_blocking=True)
        mask = batch.mask.to(self.device, non_blocking=True)
        with torch.set_grad_enabled(backward), torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            preds = self.head.score(hidden)
            loss = self.loss_policy.compute(preds, batch.targets)
            if backward and self.optimizer_policy:
                self.optimizer_policy.step(loss, lr=lr)
        return loss.item()

    def save(self, target: Union[str, Path] = "piqo", meta: Optional[WeightMeta] = None) -> Path:
        buf = io.BytesIO()
        head_sd = self.head.state_dict() if hasattr(self.head, "state_dict") else {}
        torch.save({"backbone": self.backbone.state_dict(), "head": head_sd}, buf)
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
        if isinstance(sd, dict) and "backbone" in sd and "head" in sd:
            self.backbone.load_state_dict(sd["backbone"], strict=False)
            if hasattr(self.head, "load_state_dict"):
                self.head.load_state_dict(sd["head"], strict=False)
        else:
            bb_sd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
            h_sd = {k.replace("head.", ""): v for k, v in sd.items() if k.startswith("head.")}
            if bb_sd:
                self.backbone.load_state_dict(bb_sd, strict=False)
            else:
                self.backbone.load_state_dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd, strict=False)
            if h_sd and hasattr(self.head, "load_state_dict"):
                self.head.load_state_dict(h_sd, strict=False)
