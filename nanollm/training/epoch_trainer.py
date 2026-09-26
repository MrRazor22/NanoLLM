from dataclasses import dataclass
from typing import Any, Iterator, Optional, Protocol, Union, runtime_checkable
import time
import torch
from torch.utils.data import DataLoader
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

@runtime_checkable
class ITrainer(Protocol):
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 1) -> float: ...
    def fit_iter(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 1
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

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, total_steps = torch.tensor(0.0, device=self.device), len(loader)
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                scores = self.model(ids, mask)
                loss = self.loss_fn(scores, batch["meta"], self.device) / self.accum_steps
            self.scaler.scale(loss).backward()
            step_loss = loss.detach() * self.accum_steps
            total_loss += step_loss
            if (step + 1) % self.accum_steps == 0 or (step + 1) == total_steps:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        return (total_loss / max(1, total_steps)).item()

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            for batch in loader:
                ids = batch["input_ids"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)
                scores = self.model(ids, mask)
                total_loss += self.loss_fn(scores, batch["meta"], self.device)
        return (total_loss / max(1, len(loader))).item()

    def fit_iter(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ) -> Iterator[EpochStats]:
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader) if val_loader else None
            yield EpochStats(
                epoch=epoch,
                total_epochs=epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                elapsed_sec=time.perf_counter() - t0,
            )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ) -> float:
        last_loss = 0.0
        for stats in self.fit_iter(train_loader, val_loader=val_loader, epochs=epochs):
            last_loss = stats.train_loss
        return last_loss

__all__ = ["EpochStats", "ITrainer", "EpochTrainer"]
