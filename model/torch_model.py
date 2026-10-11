from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable
import torch
import torch.nn as nn
from transformers import AutoModel

from model.layouts import Batch, ScoreLayout, TokenLayout
from model.policies.calibrated_loss import CalibratedLoss, ILoss

@runtime_checkable
class IModel(Protocol):
    """Core Primitive: Neural computation boundary for inference scoring and parameter adaptation."""
    def predict(self, layout: TokenLayout) -> ScoreLayout: ...
    def train_step(self, batch: Batch) -> float: ...
    def eval_step(self, batch: Batch) -> float: ...

class TorchModel(IModel):
    """Concrete PyTorch neural model: sovereign scoring primitive with ModernBERT backbone and scalar linear head."""

    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-base",
        checkpoint_path: Optional[Union[str, Path]] = None,
        loss_policy: Optional[ILoss] = None,
        lr: float = 1.5e-5,
        accum_steps: int = 2,
        device: Optional[torch.device] = None,
        compile: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = AutoModel.from_pretrained(backbone_name).to(self.device)
        hidden_dim = getattr(self.backbone.config, "hidden_size", 768) if hasattr(self.backbone, "config") else 768
        self.head = nn.Linear(hidden_dim, 1).to(self.device)

        if checkpoint_path:
            p = Path(checkpoint_path)
            if p.exists():
                ckpt = torch.load(str(p), map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                b_keys = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
                h_keys = {k.replace("head.", ""): v for k, v in state_dict.items() if k.startswith("head.")}
                if b_keys: self.backbone.load_state_dict(b_keys)
                if h_keys: self.head.load_state_dict(h_keys)

        self.loss_policy = loss_policy or CalibratedLoss()
        self.accum_steps, self.opt_count = accum_steps, 0
        params = list(self.backbone.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, fused=self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        if compile and hasattr(torch, "compile"):
            self.backbone = torch.compile(self.backbone)

    def _score(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(hidden).squeeze(-1)

    def predict(self, layout: TokenLayout) -> ScoreLayout:
        self.backbone.eval()
        self.head.eval()
        ids = torch.as_tensor(layout.input_ids, dtype=torch.long, device=self.device)
        mask = torch.as_tensor(layout.mask, dtype=torch.float, device=self.device)
        if ids.ndim == 1:
            ids, mask = ids.unsqueeze(0), mask.unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self._score(ids, mask)
        raw = logits.squeeze(0).cpu().tolist() if logits.ndim > 1 else logits.cpu().tolist()
        return ScoreLayout(scores=raw)

    def train_step(self, batch: Batch) -> float:
        self.backbone.train()
        self.head.train()
        ids = torch.as_tensor(batch.input_ids, dtype=torch.long, device=self.device)
        mask = torch.as_tensor(batch.mask, dtype=torch.float, device=self.device)
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self._score(ids, mask)
            loss = self.loss_policy.compute(logits, batch.targets) / self.accum_steps
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self.opt_count += 1
        if self.opt_count % self.accum_steps == 0:
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return (loss.detach() * self.accum_steps).item()

    def eval_step(self, batch: Batch) -> float:
        self.backbone.eval()
        self.head.eval()
        ids = torch.as_tensor(batch.input_ids, dtype=torch.long, device=self.device)
        mask = torch.as_tensor(batch.mask, dtype=torch.float, device=self.device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self._score(ids, mask)
            loss = self.loss_policy.compute(logits, batch.targets)
        return loss.item() if hasattr(loss, "item") else float(loss)

    def state_dict(self) -> dict:
        return {
            **{f"backbone.{k}": v for k, v in self.backbone.state_dict().items()},
            **{f"head.{k}": v for k, v in self.head.state_dict().items()},
        }

__all__ = ["IModel", "TorchModel"]
