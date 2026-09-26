from pathlib import Path
from typing import Any, Iterator, Optional
import torch
from torch.utils.data import DataLoader
from pipeline import PipelineComposable
from nanollm.training.epoch_trainer import EpochStats, ITrainer

class CheckpointingLayer(PipelineComposable, ITrainer):
    """ATA Composable Layer: transparently decorates ITrainer to save checkpoints on epoch improvement."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        output_path: Optional[str] = None,
        save_optimizer: bool = False,
    ):
        if inner is not None and not isinstance(inner, ITrainer):
            raise TypeError(f"inner must implement ITrainer, got {type(inner).__name__}")
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.save_optimizer = save_optimizer
        self.best_val_loss = float("inf")
        self.val_loader: Optional[DataLoader] = None

    def attach(self, inner: ITrainer) -> "CheckpointingLayer":
        if not isinstance(inner, ITrainer):
            raise TypeError(f"inner must implement ITrainer, got {type(inner).__name__}")
        self.inner = inner
        return self

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

    def fit_iter(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ) -> Iterator[EpochStats]:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer is not attached to an inner trainer.")
        self.val_loader = val_loader
        for stats in self.inner.fit_iter(train_loader, val_loader=val_loader, epochs=epochs):
            val_loss = stats.val_loss
            if val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.output_path:
                        self.save_checkpoint(self.output_path)
            elif self.output_path:
                self.save_checkpoint(self.output_path)
            yield stats

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

__all__ = ["CheckpointingLayer"]
