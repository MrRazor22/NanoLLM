from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Sequence, Union, runtime_checkable
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, logging

logging.set_verbosity_error()

if TYPE_CHECKING:
    from data.case import DecisionCase

from inference.questions import Choice, DecisionState, Question, Score, format_state
from model.layouts import Batch, PredictionLayout, SampleTargets, TargetSlot, TokenLayout
from model.policies.calibrated_loss import CalibratedLoss, ILoss

@runtime_checkable
class IModel(Protocol):
    """Core Primitive: Neural computation boundary for inference scoring and parameter adaptation."""
    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout: ...
    def train_step(self, cases: Sequence["DecisionCase"]) -> float: ...
    def eval_step(self, cases: Sequence["DecisionCase"]) -> float: ...

class TorchModel(nn.Module, IModel):
    """Concrete PyTorch neural model: ModernBERT backbone with tokenizer, slot layout, and scalar linear head."""

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
        self.device = torch.device(device) if isinstance(device, str) else (device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone = (backbone or AutoModel.from_pretrained(backbone_name)).to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(backbone_name)
        self.max_state_tokens = max_state_tokens

        hidden_dim = getattr(self.backbone.config, "hidden_size", 768) if hasattr(self.backbone, "config") else 768
        self.head = nn.Linear(hidden_dim, 1).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(str(checkpoint_path), map_location=self.device)
            sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.backbone.load_state_dict({k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}, strict=False)
            self.head.load_state_dict({k.replace("head.", ""): v for k, v in sd.items() if k.startswith("head.")}, strict=False)

        self.loss_policy = loss_policy or CalibratedLoss()
        self.accum_steps, self.opt_count = accum_steps, 0
        params = list(self.backbone.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, fused=self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        if compile and hasattr(torch, "compile"): self.backbone = torch.compile(self.backbone)

    def _encode_layout(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout:
        state_ids = self.tokenizer.encode(format_state(state))
        ids = state_ids[:self.max_state_tokens] if len(state_ids) > self.max_state_tokens else list(state_ids)
        slots = []
        for q in questions:
            ids.append(self.tokenizer.sep_token_id)
            ids.extend(self.tokenizer.encode(f" {q.name}: {q.instruction}" if q.instruction else f" {q.name}:"))
            if isinstance(q, Choice):
                pos = []
                opt_items = q.options.items() if isinstance(q.options, dict) else [(o, None) for o in q.options]
                for k, v in opt_items:
                    ids.append(self.tokenizer.mask_token_id)
                    pos.append(len(ids) - 1)
                    ids.extend(self.tokenizer.encode(f" {k}: {v}" if v else f" {k}"))
                slots.append(pos)
            else:
                ids.append(self.tokenizer.mask_token_id)
                slots.append([len(ids) - 1])
        return TokenLayout(input_ids=ids, mask=[1.0] * len(ids), slots=slots)

    def _collate_batch(self, cases: Sequence["DecisionCase"]) -> Batch:
        all_ids, targets = [], []
        for c in cases:
            layout = self._encode_layout(c.state, c.questions)
            sample_targets = []
            for q, pos in zip(c.questions, layout.slots):
                if isinstance(q, Choice):
                    keys = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                    idx = q.target if isinstance(q.target, int) else (keys.index(q.target) if q.target in keys else (int(q.target) if str(q.target).isdigit() else 0))
                    sample_targets.append(TargetSlot("choice", pos, idx))
                else:
                    sample_targets.append(TargetSlot("score" if isinstance(q, Score) else "noul", pos, float(q.target) if q.target is not None else 0.0))
            all_ids.append(list(layout.input_ids))
            targets.append(SampleTargets(sample_targets))

        max_len = max((len(ids) for ids in all_ids), default=0)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]
        masks = [[1.0] * len(ids) + [0.0] * (max_len - len(ids)) for ids in all_ids]
        return Batch(input_ids=padded_ids, mask=masks, targets=targets)

    def _score(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state).squeeze(-1)

    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout:
        layout = self._encode_layout(state, questions)
        self.backbone.eval()
        self.head.eval()
        ids = torch.as_tensor(layout.input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(layout.mask, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self._score(ids, mask)
        raw = logits.squeeze(0).cpu().tolist() if logits.ndim > 1 else logits.cpu().tolist()
        return PredictionLayout(scores=raw, slots=layout.slots)

    def train_step(self, cases: Sequence["DecisionCase"]) -> float:
        batch = self._collate_batch(cases)
        self.backbone.train()
        self.head.train()
        ids = torch.as_tensor(batch.input_ids, dtype=torch.long, device=self.device)
        mask = torch.as_tensor(batch.mask, dtype=torch.float, device=self.device)
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
        batch = self._collate_batch(cases)
        self.backbone.eval()
        self.head.eval()
        ids = torch.as_tensor(batch.input_ids, dtype=torch.long, device=self.device)
        mask = torch.as_tensor(batch.mask, dtype=torch.float, device=self.device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            loss = self.loss_policy.compute(self._score(ids, mask), batch.targets)
        return loss.item() if hasattr(loss, "item") else float(loss)

    def state_dict(self) -> dict:
        return {**{f"backbone.{k}": v for k, v in self.backbone.state_dict().items()}, **{f"head.{k}": v for k, v in self.head.state_dict().items()}}

__all__ = ["IModel", "TorchModel"]
