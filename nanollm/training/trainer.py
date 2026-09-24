from typing import Protocol
import torch
from torch.utils.data import DataLoader
from nanollm.engine.substrate import NanoModel
from nanollm.training.policies.loss import CalibratedLoss

class ITrainer(Protocol):
    def train_epoch(self, loader: DataLoader) -> float:
        ...
    def evaluate(self, loader: DataLoader) -> float:
        ...

class EpochTrainer:
    def __init__(
        self,
        model: NanoModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: CalibratedLoss,
        device: torch.device,
        accum_steps: int = 4,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.accum_steps = accum_steps
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, total_steps = 0.0, len(loader)
        self.optimizer.zero_grad()
        for step, batch in enumerate(loader):
            ids, mask = batch["input_ids"].to(self.device), batch["mask"].to(self.device)
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                scores = self.model(ids, mask)
                loss = self.loss_fn(scores, batch["meta"], self.device) / self.accum_steps
            self.scaler.scale(loss).backward()
            total_loss += loss.item() * self.accum_steps
            if (step + 1) % self.accum_steps == 0 or (step + 1) == total_steps:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        return total_loss / max(1, total_steps)

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            for batch in loader:
                ids, mask = batch["input_ids"].to(self.device), batch["mask"].to(self.device)
                scores = self.model(ids, mask)
                loss = self.loss_fn(scores, batch["meta"], self.device)
                total_loss += loss.item()
        return total_loss / max(1, len(loader))
