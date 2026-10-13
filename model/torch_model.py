import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Sequence, Union, runtime_checkable
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, logging

logging.set_verbosity_error()

if TYPE_CHECKING:
    from data.case import DecisionCase

from inference.questions import DecisionState, Question
from model.calibrated_loss import CalibratedLoss, ILoss
from model.checkpoints import DEFAULT_CHECKPOINT, CheckpointMeta, resolve_device
from model.layouts import Batch, PredictionLayout, collate_batch, encode_layout

@runtime_checkable
class IModel(Protocol):
    """Core Primitive: Neural computation boundary for inference scoring and parameter adaptation."""
    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout: ...
    def train_step(self, cases: Sequence["DecisionCase"]) -> float: ...
    def eval_step(self, cases: Sequence["DecisionCase"]) -> float: ...
    def save(self, path: Union[str, Path], meta: Optional[CheckpointMeta] = None) -> None: ...
    def load(self, path: Union[str, Path]) -> None: ...

class TorchModel(nn.Module, IModel):
    """Concrete PyTorch neural model: ModernBERT backbone with tokenizer, slot layout, and scalar head."""

    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-base",
        backbone: Optional[nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        loss_policy: Optional[ILoss] = None,
        lr: float = 1.5e-5,
        accum_steps: int = 2,
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
        self.backbone = (backbone or AutoModel.from_pretrained(backbone_name)).to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(backbone_name)
        self.max_state_tokens = max_state_tokens

        hidden_dim = getattr(self.backbone.config, "hidden_size", 768) if hasattr(self.backbone, "config") else 768
        self.head = nn.Linear(hidden_dim, 1).to(self.device)

        ckpt = DEFAULT_CHECKPOINT if checkpoint_path is None and DEFAULT_CHECKPOINT.exists() else checkpoint_path
        if ckpt:
            self.load(ckpt)

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
        layout = encode_layout(self.tokenizer, state, questions, max_state_tokens=self.max_state_tokens, tok_cache=self._tok_cache)
        self.backbone.eval()
        self.head.eval()
        ids = torch.as_tensor(layout.input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(layout.mask, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self._score(ids, mask)
        raw = logits.squeeze(0).cpu().tolist() if logits.ndim > 1 else logits.cpu().tolist()
        return PredictionLayout(scores=raw, slots=layout.slots)

    def train_step(self, cases: Sequence["DecisionCase"]) -> float:
        batch = collate_batch(self.tokenizer, cases, max_state_tokens=self.max_state_tokens, shuffle_options=True, tok_cache=self._tok_cache)
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

    def eval_step(self, cases: Sequence["DecisionCase"]) -> float:
        batch = collate_batch(self.tokenizer, cases, max_state_tokens=self.max_state_tokens, shuffle_options=False, tok_cache=self._tok_cache)
        self.backbone.eval()
        self.head.eval()
        ids = batch.input_ids.to(self.device, non_blocking=True)
        mask = batch.mask.to(self.device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            loss = self.loss_policy.compute(self._score(ids, mask), batch.targets)
        return loss.item() if hasattr(loss, "item") else float(loss)

    def load(self, path: Union[str, Path]) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {p}")
        ckpt = torch.load(str(p), map_location=self.device)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.backbone.load_state_dict({k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}, strict=False)
        self.head.load_state_dict({k.replace("head.", ""): v for k, v in sd.items() if k.startswith("head.")}, strict=False)

    def save(self, path: Union[str, Path], meta: Optional[CheckpointMeta] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sd = {**{f"backbone.{k}": v for k, v in self.backbone.state_dict().items()}, **{f"head.{k}": v for k, v in self.head.state_dict().items()}}
        meta_dict = {"epoch": meta.epoch, "train_loss": meta.train_loss, "val_loss": meta.val_loss, "saved_at": meta.saved_at} if meta else {}
        torch.save({"model_state_dict": sd, "meta": meta_dict}, str(p))
        manifest_p = p.parent / "manifest.json"
        manifest = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {}
        manifest[p.name] = meta_dict
        manifest_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

__all__ = ["IModel", "TorchModel"]
