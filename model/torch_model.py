from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, Union, runtime_checkable
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, logging

logging.set_verbosity_error()

if TYPE_CHECKING:
    from data.case import DecisionCase

from inference.questions import DecisionState, Question
from model.model_extensions import resolve_device
from model.policies.calibrated_loss import CalibratedLoss, ILoss
from model.policies.layout_policy import ILayoutPolicy, MaskedLayoutPolicy
from model.policies.weight_cache import DiskWeightCache, IWeightCache
from model.token_layout import Batch, PredictionLayout
from model.weights import Weights

@runtime_checkable
class IModel(Protocol):
    """Core Primitive: Neural computation boundary for inference scoring and parameter adaptation."""
    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout: ...
    def train_step(self, cases: Sequence["DecisionCase"], lr: Optional[float] = None) -> float: ...
    def eval_step(self, cases: Sequence["DecisionCase"]) -> float: ...
    def export(self) -> Weights: ...
    def load(self, source: Optional[Union[Weights, str, Path]] = None) -> None: ...
    def save(self, target: Union[str, Path] = "piqo", stats: Optional[Any] = None) -> Path: ...

class TorchModel(nn.Module, IModel):
    """Core Primitive Implementation: ModernBERT backbone with scalar classification head."""

    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-base",
        backbone: Optional[nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        weights: Optional[Weights] = None,
        loss_policy: Optional[ILoss] = None,
        weight_cache: Optional[IWeightCache] = None,
        layout_policy: Optional[ILayoutPolicy] = None,
        lr: float = 2e-5,
        accum_steps: int = 1,
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
        self.weight_cache = weight_cache or DiskWeightCache()
        self.layout_policy = layout_policy or MaskedLayoutPolicy()
        self.loss_policy = loss_policy or CalibratedLoss()
        self.backbone = (backbone or AutoModel.from_pretrained(backbone_name, **model_kwargs)).to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(backbone_name)
        self.max_state_tokens = max_state_tokens
        self.accum_steps, self.opt_count = max(1, accum_steps), 0
        self._tok_cache: dict[str, list[int]] = {}

        hidden_dim = getattr(self.backbone.config, "hidden_size", 768) if hasattr(self.backbone, "config") else 768
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        params = list(self.backbone.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, fused=self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        if weights is not None:
            self.load(weights)
        if compile and hasattr(torch, "compile"):
            self.backbone = torch.compile(self.backbone)

    def _score(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(hidden).squeeze(-1)

    def _forward(self, batch: Batch) -> torch.Tensor:
        ids = batch.input_ids.to(self.device, non_blocking=True)
        mask = batch.mask.to(self.device, non_blocking=True)
        return self._score(ids, mask)

    def _collate(self, cases: Sequence["DecisionCase"], shuffle_options: bool) -> Batch:
        return self.layout_policy.collate(
            self.tokenizer, cases, max_state_tokens=self.max_state_tokens,
            shuffle_options=shuffle_options, tok_cache=self._tok_cache,
        )

    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout:
        layout = self.layout_policy.encode(self.tokenizer, state, questions, max_state_tokens=self.max_state_tokens, tok_cache=self._tok_cache)
        self.eval()
        ids = torch.as_tensor(layout.input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(layout.mask, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            logits = self._score(ids, mask)
        scores = logits.squeeze(0).cpu().tolist() if logits.ndim > 1 else logits.cpu().tolist()
        return PredictionLayout(scores=scores, slots=layout.slots)

    def train_step(self, cases: Sequence["DecisionCase"], lr: Optional[float] = None) -> float:
        if lr is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = lr
        batch = self._collate(cases, shuffle_options=True)
        self.train()
        with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            loss = self.loss_policy.compute(self._forward(batch), batch.targets) / self.accum_steps
        (self.scaler.scale(loss) if self.scaler.is_enabled() else loss).backward()
        self.opt_count += 1
        if self.opt_count % self.accum_steps == 0:
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return (loss.detach() * self.accum_steps).item()

    def eval_step(self, cases: Sequence["DecisionCase"]) -> float:
        batch = self._collate(cases, shuffle_options=False)
        self.eval()
        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            loss = self.loss_policy.compute(self._forward(batch), batch.targets)
        return loss.item()

    def export(self) -> Weights:
        import io
        buf = io.BytesIO()
        sd = {**{f"backbone.{k}": v for k, v in self.backbone.state_dict().items()}, **{f"head.{k}": v for k, v in self.head.state_dict().items()}}
        torch.save({"model_state_dict": sd}, buf)
        return Weights(payload=buf.getvalue())

    def load(self, source: Optional[Union[Weights, str, Path]] = None) -> None:
        target = "piqo" if source is None else source
        if isinstance(target, Weights):
            weights = target
        else:
            w = self.weight_cache.get(target)
            if not w:
                raise FileNotFoundError(f"Weights not found in cache: {target}")
            weights = w
        if not weights.payload:
            return
        import io
        data = torch.load(io.BytesIO(weights.payload), map_location=self.device)
        sd = data.get("model_state_dict", data) if isinstance(data, dict) and "model_state_dict" in data else data
        self.backbone.load_state_dict({k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}, strict=False)
        self.head.load_state_dict({k.replace("head.", ""): v for k, v in sd.items() if k.startswith("head.")}, strict=False)

    def save(self, target: Union[str, Path] = "piqo", stats: Optional[Any] = None) -> Path:
        exported = self.export()
        import time
        w = Weights.from_bytes(
            payload=exported.payload,
            name=Path(target).stem,
            epoch=getattr(stats, "epoch", 1) if stats else 1,
            train_loss=round(stats.loss, 6) if stats and hasattr(stats, "loss") else None,
            val_loss=round(stats.val_loss, 6) if stats and getattr(stats, "val_loss", None) is not None else None,
            saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return self.weight_cache.put(target, w)
