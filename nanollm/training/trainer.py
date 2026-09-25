from typing import Any, Optional, Protocol, Tuple
import time
import torch
from torch.utils.data import DataLoader
from nanollm.model import NanoModel
from nanollm.training.policies.curriculum import ICurriculum
from nanollm.training.policies.dataset import to_decision_sample
from nanollm.training.policies.loss import CalibratedLoss

class ITrainer(Protocol):
    def train_epoch(self, loader: DataLoader) -> float: ...
    def evaluate(self, loader: DataLoader) -> float: ...
    def add(self, layer: Any, **kwargs: Any) -> "ITrainer": ...
    def __or__(self, layer: Any) -> "ITrainer": ...

class EpochTrainer(ITrainer):
    def add(self, layer: Any, **kwargs: Any) -> "ITrainer":
        if isinstance(layer, type):
            return layer(self, **kwargs)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer(self, **kwargs)

    def __or__(self, layer: Any) -> "ITrainer":
        return self.add(layer)
    def __init__(
        self,
        model: NanoModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: CalibratedLoss,
        device: torch.device,
        curriculum: Optional[ICurriculum] = None,
        accum_steps: int = 4,
        log_interval: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.curriculum = curriculum
        self.accum_steps = accum_steps
        self.log_interval = log_interval
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def build_dataloaders(self, collator: Any, batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
        if self.curriculum is None:
            raise ValueError("No curriculum policy injected into EpochTrainer")
        train_raw, val_raw = self.curriculum.build()
        train_samples = [to_decision_sample(r) for r in train_raw]
        val_samples = [to_decision_sample(r) for r in val_raw]
        train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=False, collate_fn=collator)
        return train_loader, val_loader

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, total_steps = 0.0, len(loader)
        start_time = time.perf_counter()
        self.optimizer.zero_grad()
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                scores = self.model(ids, mask)
                loss = self.loss_fn(scores, batch["meta"], self.device) / self.accum_steps
            self.scaler.scale(loss).backward()
            total_loss += loss.item() * self.accum_steps
            if (step + 1) % self.accum_steps == 0 or (step + 1) == total_steps:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            log_int = self.log_interval if self.log_interval > 0 else min(100, max(1, total_steps // 10))
            if (step + 1) % log_int == 0 or (step + 1) == total_steps:
                elapsed = time.perf_counter() - start_time
                avg_step_ms = (elapsed / (step + 1)) * 1000.0
                curr_loss = total_loss / (step + 1)
                print(
                    f"Step [{step+1:5d}/{total_steps}] Loss: {curr_loss:.4f} | Speed: {avg_step_ms:.1f}ms/step",
                    flush=True
                )
        return total_loss / max(1, total_steps)

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            for batch in loader:
                ids = batch["input_ids"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)
                scores = self.model(ids, mask)
                loss = self.loss_fn(scores, batch["meta"], self.device)
                total_loss += loss.item()
        return total_loss / max(1, len(loader))
