from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, Union, runtime_checkable
import time
import torch
from pipeline import PipelineComposable
from nanollm.model import NanoModel
from nanollm.training.loss import CalibratedLoss, ILoss

@dataclass(frozen=True)
class StepStats:
    epoch: int
    total_epochs: int
    step: int
    total_steps: int  # 0 = unknown
    loss: float
    elapsed_ms: float
    lr: float = 0.0
    val_loss: Optional[float] = None  # set on final step of each epoch
    batch: int = 0
    total_batches: int = 0

Batch = Mapping[str, Any]

@runtime_checkable
class ITrainer(Protocol):
    """ATA Root Contract: fit streams one StepStats per optimizer step."""
    def fit(
        self,
        train_batches: Iterable[Batch],
        val_batches: Optional[Iterable[Batch]] = None,
        epochs: int = 1,
    ) -> Iterator[StepStats]: ...

    def evaluate(self, batches: Iterable[Batch]) -> float: ...

class EpochTrainer(PipelineComposable, ITrainer):
    def __init__(
        self,
        model: NanoModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[CalibratedLoss, ILoss],
        device: torch.device,
        accum_steps: int = 2,
    ):
        if not isinstance(loss_fn, ILoss):
            raise TypeError(f"loss_fn must implement ILoss, got {type(loss_fn).__name__}")
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.accum_steps = accum_steps
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    @classmethod
    def from_backbone(
        cls,
        backbone_name: str = "answerdotai/ModernBERT-base",
        lr: float = 1.5e-5,
        device: Optional[torch.device] = None,
        accum_steps: int = 2,
        init_checkpoint: Optional[str] = None,
    ) -> "EpochTrainer":
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NanoModel.from_backbone(backbone_name).to(dev)
        if init_checkpoint:
            ckpt = torch.load(init_checkpoint, map_location=dev)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=dev.type == "cuda")
        return cls(model, optimizer, CalibratedLoss(), dev, accum_steps=accum_steps)

    def evaluate(self, batches: Iterable[Batch]) -> float:
        self.model.eval()
        val_total, m = torch.tensor(0.0, device=self.device), 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            for m, batch in enumerate(batches):
                ids = batch["input_ids"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)
                val_total += self.loss_fn(self.model(ids, mask), batch["meta"], self.device)
        return (val_total / max(1, m + 1)).item()

    def fit(
        self,
        train_batches: Iterable[Batch],
        val_batches: Optional[Iterable[Batch]] = None,
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
                ids = batch["input_ids"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    loss = self.loss_fn(self.model(ids, mask), batch["meta"], self.device) / self.accum_steps
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
            # flush leftover accumulation if total batches not divisible by accum_steps
            remainder = (len(list(range(0))) if not total_batches else total_batches) % self.accum_steps
            if total_batches and total_batches % self.accum_steps != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                cur_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
                val_loss = self.evaluate(val_batches) if val_batches is not None else None
                yield StepStats(epoch, epochs, opt_step, total_steps or opt_step, (total_loss / max(1, total_batches)).item(), 0.0, cur_lr, val_loss, batch=total_batches, total_batches=total_batches)

__all__ = ["StepStats", "ITrainer", "EpochTrainer", "Batch"]
