from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, Union, runtime_checkable
import time
import torch
from pipeline import PipelineComposable
from nanollm.model import NanoModel
from nanollm.training.loss import CalibratedLoss, ILoss

@dataclass(frozen=True)
class EpochStats:
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: Optional[float]
    elapsed_sec: float

Batch = Mapping[str, Any]

@runtime_checkable
class ITrainer(Protocol):
    def train_epoch(self, batches: Iterable[Batch]) -> float: ...
    def evaluate(self, batches: Iterable[Batch]) -> float: ...
    def fit(
        self,
        train_batches: Iterable[Batch],
        val_batches: Optional[Iterable[Batch]] = None,
        epochs: int = 1,
    ) -> Iterator[EpochStats]: ...

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
            model.load_state_dict(torch.load(init_checkpoint, map_location=dev))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=dev.type == "cuda")
        loss_fn = CalibratedLoss()
        return cls(model, optimizer, loss_fn, dev, accum_steps=accum_steps)

    def train_epoch(self, batches: Iterable[Batch]) -> float:
        self.model.train()
        total_loss, steps = torch.tensor(0.0, device=self.device), 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(batches):
            steps += 1
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                scores = self.model(ids, mask)
                loss = self.loss_fn(scores, batch["meta"], self.device) / self.accum_steps
            self.scaler.scale(loss).backward()
            total_loss += loss.detach() * self.accum_steps
            if (step + 1) % self.accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        if steps % self.accum_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return (total_loss / max(1, steps)).item()

    def evaluate(self, batches: Iterable[Batch]) -> float:
        self.model.eval()
        total_loss, steps = torch.tensor(0.0, device=self.device), 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            for batch in batches:
                steps += 1
                ids = batch["input_ids"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)
                scores = self.model(ids, mask)
                total_loss += self.loss_fn(scores, batch["meta"], self.device)
        return (total_loss / max(1, steps)).item()

    def fit(
        self,
        train_batches: Iterable[Batch],
        val_batches: Optional[Iterable[Batch]] = None,
        epochs: int = 1,
    ) -> Iterator[EpochStats]:
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss = self.train_epoch(train_batches)
            val_loss = self.evaluate(val_batches) if val_batches else None
            yield EpochStats(
                epoch=epoch,
                total_epochs=epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                elapsed_sec=time.perf_counter() - t0,
            )

__all__ = ["EpochStats", "ITrainer", "EpochTrainer", "Batch"]
