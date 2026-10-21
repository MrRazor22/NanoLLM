from pathlib import Path
import time
from typing import Optional, Protocol, Sequence, Union, runtime_checkable
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, logging

logging.set_verbosity_error()

from data.case import DecisionCase
from model.device import resolve_device
from model.policies.head import IHead, MLPHead
from model.policies.layout import ILayout, MaskedLayout
from model.policies.weight_cache import DiskWeightCache, IWeightCache
from model.weights import WeightMeta, Weights
from training.policies.learner import AdamWLearner, ILearner

@runtime_checkable
class IModel(Protocol):
    def forward(self, cases: Sequence[DecisionCase], train: bool = False) -> torch.Tensor: ...
    def step(self, loss: torch.Tensor, lr: Optional[float] = None, commit: bool = True) -> None: ...
    def save(self, target: Union[str, Path] = "piqo", meta: Optional[WeightMeta] = None) -> Path: ...
    def load(self, source: Optional[Union[Weights, str, Path]] = None, strict: bool = True) -> None: ...

class TorchModel(IModel):
    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-base",
        backbone: Optional[nn.Module] = None,
        head: Optional[IHead] = None,
        layout: Optional[ILayout] = None,
        learner: Optional[ILearner] = None,
        weight_cache: Optional[IWeightCache] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        weights: Optional[Union[Weights, str, Path]] = None,
        lr: float = 2e-5,
        accum_steps: int = 1,
        max_state_tokens: int = 2048,
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
        self.weight_cache = weight_cache or DiskWeightCache()

        parameters = list(self.backbone.parameters()) + list(self.head.parameters())
        self.learner = learner or AdamWLearner(parameters, lr=lr, accum_steps=accum_steps, device=self.device)

        if weights is not None:
            self.load(weights)
        if compile and hasattr(torch, "compile"):
            self.backbone = torch.compile(self.backbone)

    def forward(self, cases: Sequence[DecisionCase], train: bool = False) -> torch.Tensor:
        prepared = [c.shuffled() if train else c for c in cases]
        layouts = [self.layout.layout(c) for c in prepared]
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        max_len = max(len(l.tokens) for l in layouts)

        input_ids = torch.full((len(layouts), max_len), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(layouts), max_len), dtype=torch.float, device=self.device)
        for i, l in enumerate(layouts):
            sz = len(l.tokens)
            input_ids[i, :sz] = torch.as_tensor(l.tokens, dtype=torch.long, device=self.device)
            mask[i, :sz] = 1.0

        self.backbone.train(train)
        self.head.train(train)
        with torch.set_grad_enabled(train), torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            hidden = self.backbone(input_ids=input_ids, attention_mask=mask).last_hidden_state
            preds = self.head.score(hidden)
            candidate_scores = pad_sequence([preds[b, l.slots] for b, l in enumerate(layouts)], batch_first=True, padding_value=-1e4)
        return candidate_scores

    def step(self, loss: torch.Tensor, lr: Optional[float] = None, commit: bool = True) -> None:
        self.learner.step(loss, lr=lr, commit=commit)

    def save(self, target: Union[str, Path] = "piqo", meta: Optional[WeightMeta] = None) -> Path:
        from safetensors.torch import save
        tensors = {f"backbone.{k}": v.contiguous() for k, v in self.backbone.state_dict().items()}
        tensors.update({f"head.{k}": v.contiguous() for k, v in self.head.state_dict().items()})
        raw_bytes = save(tensors)
        metadata = meta or WeightMeta(saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        if not metadata.saved_at:
            metadata = WeightMeta(epoch=metadata.epoch, train_loss=metadata.train_loss, val_loss=metadata.val_loss, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        return self.weight_cache.put(target, Weights(payload=raw_bytes, name=Path(target).stem, meta=metadata))

    def load(self, source: Optional[Union[Weights, str, Path]] = None, strict: bool = True) -> None:
        target = "piqo" if source is None else source
        weights = target if isinstance(target, Weights) else self.weight_cache.get(target)
        if not weights:
            raise FileNotFoundError(f"Weights not found: {target}")
        if not weights.payload:
            return
        from safetensors.torch import load
        sd = load(weights.payload)
        bb_sd = {k.removeprefix("backbone."): v for k, v in sd.items() if k.startswith("backbone.")}
        h_sd = {k.removeprefix("head."): v for k, v in sd.items() if k.startswith("head.")}
        if bb_sd:
            res_bb = self.backbone.load_state_dict(bb_sd, strict=strict)
            if not strict and (res_bb.missing_keys or res_bb.unexpected_keys):
                import sys
                sys.stderr.write(f"\n[CHECKPOINT WARNING] '{target}' backbone missing: {res_bb.missing_keys}, unexpected: {res_bb.unexpected_keys}\n")
        if h_sd:
            res_h = self.head.load_state_dict(h_sd, strict=strict)
            if not strict and (res_h.missing_keys or res_h.unexpected_keys):
                import sys
                sys.stderr.write(f"\n[CHECKPOINT WARNING] '{target}' head missing: {res_h.missing_keys}, unexpected: {res_h.unexpected_keys}\n")
