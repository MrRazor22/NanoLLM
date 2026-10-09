import time
from typing import Iterable, Iterator, Optional, Protocol, Union, runtime_checkable
import torch
import torch.nn as nn
from transformers import AutoModel

from nanollm.engine.policies.torch_scorer import TorchScorer, _ScoringHead
from nanollm.trainer.layers.checkpointing_layer import CheckpointingLayer
from nanollm.trainer.batches import StepStats, TrainingBatch
from nanollm.trainer.policies.calibrated_loss import CalibratedLoss, ILoss
from nanollm.trainer.layers.metrics_layer import MetricsLayer

# ==========================================
# Core Primitive (P)
# (Pure Contract: Zero torch, Zero Any)
# ==========================================

@runtime_checkable
class ITrainer(Protocol):
    """Core Primitive: Operational boundary for model optimization and evaluation."""

    def fit(
        self,
        train_batches: Iterable[TrainingBatch],
        val_batches: Optional[Iterable[TrainingBatch]] = None,
        epochs: int = 1,
    ) -> Iterator[StepStats]:
        """Streams real-time step statistics across training epochs."""
        ...

    def evaluate(self, batches: Iterable[TrainingBatch]) -> float:
        """Evaluates average loss over the provided batches."""
        ...

class EpochTrainer(ITrainer):
    """Core Primitive Implementation: Supervised training loop over assembled batches."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[ILoss] = None,
        device: Optional[torch.device] = None,
        accum_steps: int = 2,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn or CalibratedLoss()
        self.accum_steps = accum_steps
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

    @classmethod
    def from_backbone(
        cls,
        backbone_name: str = "answerdotai/ModernBERT-base",
        lr: float = 1.5e-5,
        device: Optional[torch.device] = None,
        accum_steps: int = 2,
        init_checkpoint: Optional[str] = None,
        compile: bool = False,
    ) -> "EpochTrainer":
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scorer = TorchScorer.from_backbone(backbone_name, checkpoint_path=init_checkpoint, device=dev, compile=compile)
        optimizer = torch.optim.AdamW(scorer.model.parameters(), lr=lr, fused=dev.type == "cuda")
        return cls(scorer.model, optimizer, device=dev, accum_steps=accum_steps)

    def evaluate(self, batches: Iterable[TrainingBatch]) -> float:
        self.model.eval()
        val_total, m = torch.tensor(0.0, device=self.device), 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            for m, batch in enumerate(batches):
                raw_ids = batch["input_ids"] if isinstance(batch, dict) or hasattr(batch, "__getitem__") else batch.input_ids
                raw_mask = batch["mask"] if isinstance(batch, dict) or hasattr(batch, "__getitem__") else batch.mask
                meta = batch["targets"] if (isinstance(batch, dict) or hasattr(batch, "__getitem__")) and "targets" in batch else (batch.targets if hasattr(batch, "targets") else batch["meta"])
                ids = torch.as_tensor(raw_ids, dtype=torch.long, device=self.device)
                mask = torch.as_tensor(raw_mask, dtype=torch.float, device=self.device)
                scores = self.model(ids, mask)
                loss = self.loss_fn.loss_tensor(scores, meta) if hasattr(self.loss_fn, "loss_tensor") else self.loss_fn.forward(scores, meta)
                val_total += loss
        return (val_total / max(1, m + 1)).item()

    def fit(
        self,
        train_batches: Iterable[TrainingBatch],
        val_batches: Optional[Iterable[TrainingBatch]] = None,
        epochs: int = 1,
    ) -> Iterator[StepStats]:
        total_batches = len(train_batches) if hasattr(train_batches, "__len__") else 0
        total_steps = total_batches // self.accum_steps if total_batches else 0
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = torch.tensor(0.0, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            last_t = time.perf_counter()
            opt_step = 0
            for n, batch in enumerate(train_batches):
                raw_ids = batch["input_ids"] if isinstance(batch, dict) or hasattr(batch, "__getitem__") else batch.input_ids
                raw_mask = batch["mask"] if isinstance(batch, dict) or hasattr(batch, "__getitem__") else batch.mask
                meta = batch["targets"] if (isinstance(batch, dict) or hasattr(batch, "__getitem__")) and "targets" in batch else (batch.targets if hasattr(batch, "targets") else batch["meta"])
                ids = torch.as_tensor(raw_ids, dtype=torch.long, device=self.device)
                mask = torch.as_tensor(raw_mask, dtype=torch.float, device=self.device)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    scores = self.model(ids, mask)
                    loss_node = self.loss_fn.loss_tensor(scores, meta) if hasattr(self.loss_fn, "loss_tensor") else self.loss_fn.forward(scores, meta)
                    loss = loss_node / self.accum_steps
                self.scaler.scale(loss).backward()
                total_loss += loss.detach() * self.accum_steps
                if (n + 1) % self.accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    opt_step += 1
                    now = time.perf_counter()
                    is_last = (total_steps > 0 and opt_step >= total_steps)
                    val_loss = self.evaluate(val_batches) if (is_last and val_batches is not None) else None
                    cur_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
                    yield StepStats(epoch, epochs, opt_step, total_steps, (total_loss / (n + 1)).item(), (now - last_t) * 1000.0 / self.accum_steps, cur_lr, val_loss, batch=n + 1, total_batches=total_batches)
                    last_t = now
            if total_batches and total_batches % self.accum_steps != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                cur_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
                val_loss = self.evaluate(val_batches) if val_batches is not None else None
                yield StepStats(epoch, epochs, opt_step, total_steps or opt_step, (total_loss / max(1, total_batches)).item(), 0.0, cur_lr, val_loss, batch=total_batches, total_batches=total_batches)

    def add(self, layer: Union[type, CheckpointingLayer, MetricsLayer]) -> ITrainer:
        if isinstance(layer, type):
            return layer(self)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer

    def __or__(self, layer: Union[type, CheckpointingLayer, MetricsLayer]) -> ITrainer:
        return self.add(layer)

__all__ = ["ITrainer", "EpochTrainer"]
