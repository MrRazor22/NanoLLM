from pathlib import Path
from typing import Any, Optional
import torch
from torch.utils.data import DataLoader
from nanollm.training.trainer import ITrainer

class CheckpointingLayer(ITrainer):
    """ATA Composable Layer: transparently decorates ITrainer to save checkpoints on epoch improvement."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        output_path: Optional[str] = None,
        save_optimizer: bool = False,
    ):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.save_optimizer = save_optimizer
        self.best_val_loss = float("inf")
        self.val_loader: Optional[DataLoader] = None

    def attach(self, inner: ITrainer) -> "CheckpointingLayer":
        self.inner = inner
        return self

    def add(self, layer: Any, **kwargs: Any) -> "ITrainer":
        if isinstance(layer, type):
            return layer(self, **kwargs)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer(self, **kwargs)

    def __or__(self, layer: Any) -> "ITrainer":
        return self.add(layer)

    @property
    def model(self) -> Any:
        return getattr(self.inner, "model")

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_state_dict": self.model.state_dict()}
        if self.save_optimizer and hasattr(self.inner, "optimizer"):
            payload["optimizer_state_dict"] = getattr(self.inner, "optimizer").state_dict()
        torch.save(payload, str(path))

    def train_epoch(self, loader: DataLoader) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer is not attached to an inner trainer.")
        loss = self.inner.train_epoch(loader)
        if self.val_loader:
            val_loss = self.inner.evaluate(self.val_loader)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.output_path:
                    self.save_checkpoint(self.output_path)
        elif self.output_path:
            self.save_checkpoint(self.output_path)
        return loss

    def evaluate(self, loader: DataLoader) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer is not attached to an inner trainer.")
        return self.inner.evaluate(loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer is not attached to an inner trainer.")
        self.val_loader = val_loader

        last_loss = 0.0
        for epoch in range(1, epochs + 1):
            last_loss = self.train_epoch(train_loader)
            val_info = f" | Best Val Loss: {self.best_val_loss:.4f}" if self.val_loader else ""
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {last_loss:.4f}{val_info}")
        return last_loss

__all__ = ["CheckpointingLayer"]
