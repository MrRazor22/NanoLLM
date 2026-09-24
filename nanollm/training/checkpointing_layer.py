from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
from nanollm.training.trainer import ITrainer

class CheckpointingLayer:
    def __init__(
        self,
        inner: ITrainer,
        output_path: str,
        val_loader: Optional[DataLoader] = None,
    ):
        self.inner = inner
        self.output_path = output_path
        self.val_loader = val_loader
        self.best_val_loss = float("inf")

    def train_epoch(self, loader: DataLoader) -> float:
        train_loss = self.inner.train_epoch(loader)
        if self.val_loader is not None:
            val_loss = self.inner.evaluate(self.val_loader)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                p = Path(self.output_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                if hasattr(self.inner, "model"):
                    torch.save(self.inner.model.state_dict(), str(p))
        return train_loss

    def evaluate(self, loader: DataLoader) -> float:
        return self.inner.evaluate(loader)

CheckpointingTrainer = CheckpointingLayer
