from typing import TYPE_CHECKING, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable
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
    def train_step(self, cases: Sequence["DecisionCase"]) -> float: ...
    def eval_step(self, cases: Sequence["DecisionCase"]) -> float: ...
    def export(self) -> Weights: ...
    def load(self, source: Optional[Union[Weights, str, Path]] = None) -> None: ...
    def save(self, target: Union[str, Path] = "piqo", stats: Optional[Any] = None) -> Path: ...

class TorchModel(nn.Module, IModel):
    """Concrete PyTorch neural model: ModernBERT backbone with tokenizer, slot layout, and scalar head."""

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
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
        model_kwargs = {"attn_implementation": "sdpa"} if self.device.type == "cuda" else {}
        self.weight_cache = weight_cache or DiskWeightCache()
        self.layout_policy = layout_policy or MaskedLayoutPolicy()
        self.backbone = (backbone or AutoModel.from_pretrained(backbone_name, **model_kwargs)).to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(backbone_name)
        self.max_state_tokens = max_state_tokens

        hidden_dim = getattr(self.backbone.config, "hidden_size", 768) if hasattr(self.backbone, "config") else 768
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)
        nn.init.xavier_uniform_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)
        nn.init.xavier_uniform_(self.head[4].weight)
        nn.init.zeros_(self.head[4].bias)

        if weights is not None:
            self.load(weights)

        self.loss_policy = loss_policy or CalibratedLoss()
        self.accum_steps, self.opt_count = accum_steps, 0
        self._tok_cache: dict[str, list[int]] = {}
        params = list(self.backbone.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, fused=self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        if compile and hasattr(torch, "compile"): self.backbone = torch.compile(self.backbone)

    def _score(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state).squeeze(-1)

    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout:
        layout = self.layout_policy.encode(self.tokenizer, state, questions, max_state_tokens=self.max_state_tokens, tok_cache=self._tok_cache)
        self.backbone.eval()
        self.head.eval()
        ids = torch.as_tensor(layout.input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(layout.mask, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self._score(ids, mask)
        raw = logits.squeeze(0).cpu().tolist() if logits.ndim > 1 else logits.cpu().tolist()
        return PredictionLayout(scores=raw, slots=layout.slots)

    def train_step(self, cases: Union[Batch, Sequence["DecisionCase"]]) -> float:
        batch = cases if isinstance(cases, Batch) else self.layout_policy.collate(self.tokenizer, cases, max_state_tokens=self.max_state_tokens, shuffle_options=True, tok_cache=self._tok_cache)
        self.backbone.train()
        self.head.train()
        ids = batch.input_ids.to(self.device, non_blocking=True)
        mask = batch.mask.to(self.device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            loss = self.loss_policy.compute(self._score(ids, mask), batch.targets) / self.accum_steps
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

    def eval_step(self, cases: Union[Batch, Sequence["DecisionCase"]]) -> float:
        batch = cases if isinstance(cases, Batch) else self.layout_policy.collate(self.tokenizer, cases, max_state_tokens=self.max_state_tokens, shuffle_options=False, tok_cache=self._tok_cache)
        self.backbone.eval()
        self.head.eval()
        ids = batch.input_ids.to(self.device, non_blocking=True)
        mask = batch.mask.to(self.device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            loss = self.loss_policy.compute(self._score(ids, mask), batch.targets)
        return loss.item() if hasattr(loss, "item") else float(loss)

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
            epoch=stats.epoch if stats and hasattr(stats, "epoch") else 1,
            train_loss=round(stats.loss, 6) if stats and hasattr(stats, "loss") else None,
            val_loss=round(stats.val_loss, 6) if stats and getattr(stats, "val_loss", None) is not None else None,
            saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return self.weight_cache.put(target, w)
