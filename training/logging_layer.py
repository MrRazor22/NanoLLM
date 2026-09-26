from typing import Any, Optional
import time
from torch.utils.data import DataLoader
from nanollm.training.trainer import ITrainer

class LoggingLayer(ITrainer):
    """ATA Composable Layer: decorates ITrainer to provide progress and loss logging."""

    def __init__(self, inner: Optional[ITrainer] = None, prefix: str = ""):
        self.inner = inner
        self.prefix = prefix

    def attach(self, inner: ITrainer) -> "LoggingLayer":
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

    def train_epoch(self, loader: DataLoader) -> float:
        if self.inner is None:
            raise RuntimeError("LoggingLayer is not attached to an inner trainer.")
        return self.inner.train_epoch(loader)

    def evaluate(self, loader: DataLoader) -> float:
        if self.inner is None:
            raise RuntimeError("LoggingLayer is not attached to an inner trainer.")
        return self.inner.evaluate(loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ) -> float:
        if self.inner is None:
            raise RuntimeError("LoggingLayer is not attached to an inner trainer.")
        last_loss = 0.0
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            last_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader) if val_loader else None
            elapsed = time.perf_counter() - t0
            val_info = f" | Val Loss: {val_loss:.4f}" if val_loss is not None else ""
            prefix_info = f"[{self.prefix}] " if self.prefix else ""
            print(f"{prefix_info}Epoch {epoch:2d}/{epochs:2d} | Train Loss: {last_loss:.4f}{val_info} | Elapsed: {elapsed:.1f}s")
        return last_loss

__all__ = ["LoggingLayer"]
