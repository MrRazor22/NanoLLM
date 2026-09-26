from typing import Any, Optional, Protocol
import time
import torch
from torch.utils.data import DataLoader
from nanollm.model import NanoModel
from nanollm.training.loss_policy import CalibratedLoss

class ITrainer(Protocol):
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 1) -> float: ...

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
        accum_steps: int = 2,
        log_interval: float = 10.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.accum_steps = accum_steps
        self.log_interval = log_interval
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, total_steps = torch.tensor(0.0, device=self.device), len(loader)
        start_time, last_log_time = time.perf_counter(), time.perf_counter()
        interval_loss, interval_steps = torch.tensor(0.0, device=self.device), 0
        interval_sec = float(self.log_interval) if self.log_interval > 0 else 10.0
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
            interval_loss += step_loss
            interval_steps += 1
            if (step + 1) % self.accum_steps == 0 or (step + 1) == total_steps:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            now = time.perf_counter()
            if (now - last_log_time >= interval_sec) or (step + 1 == total_steps):
                step_ms = ((now - last_log_time) / max(1, interval_steps)) * 1000.0
                win_loss = (interval_loss / max(1, interval_steps)).item()
                rem_steps = total_steps - (step + 1)
                eta_sec = rem_steps * (step_ms / 1000.0)
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s" if eta_sec >= 60 else f"{int(eta_sec)}s"
                mem_str = f" | VRAM: {torch.cuda.memory_allocated(self.device)/1e9:.1f}GB" if self.device.type == "cuda" else ""
                lr_val = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Step [{step+1:5d}/{total_steps}] Loss: {win_loss:.4f} | Speed: {step_ms:.1f}ms/step | ETA: {eta_str} | LR: {lr_val:.1e}{mem_str}",
                    flush=True
                )
                last_log_time, interval_loss, interval_steps = now, torch.tensor(0.0, device=self.device), 0
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

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ) -> float:
        last_loss = 0.0
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            last_loss = self.train_epoch(train_loader)
            val_info = f" | Val Loss: {self.evaluate(val_loader):.4f}" if val_loader else ""
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {last_loss:.4f}{val_info} | Elapsed: {time.perf_counter() - t0:.1f}s")
        return last_loss

__all__ = ["ITrainer", "EpochTrainer"]
