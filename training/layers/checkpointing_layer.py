from pathlib import Path
from typing import Any, Optional, Union
import torch
from torch.utils.data import DataLoader
from nanollm.training.trainer import ITrainer

class CheckpointingLayer(ITrainer):
    def __init__(
        self,
        inner_or_output_path: Union[ITrainer, str, None] = None,
        output_path: Optional[str] = None,
        val_loader: Optional[DataLoader] = None,
        inner: Optional[ITrainer] = None,
    ):
        if isinstance(inner_or_output_path, str):
            self.inner = inner
            self.output_path = inner_or_output_path
        elif inner_or_output_path is not None:
            self.inner = inner_or_output_path
            self.output_path = output_path or ""
        else:
            self.inner = inner
            self.output_path = output_path or ""
        self.val_loader = val_loader
        self.best_val_loss = float("inf")

    def attach(self, inner: ITrainer) -> "CheckpointingLayer":
        self.inner = inner
        return self

    def add(self, layer: Any, **kwargs: Any) -> ITrainer:
        if isinstance(layer, type):
            return layer(self, **kwargs)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer(self, **kwargs)

    def __or__(self, layer: Any) -> ITrainer:
        return self.add(layer)

    def train_epoch(self, loader: DataLoader) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer is not attached to an inner trainer.")
        train_loss = self.inner.train_epoch(loader)
        if self.val_loader is not None:
            val_loss = self.inner.evaluate(self.val_loader)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                p = Path(self.output_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                if hasattr(self.inner, "model"):
                    torch.save(self.inner.model.state_dict(), str(p))
                    print(f"Checkpoint saved -> {p} (Val Loss: {val_loss:.4f})", flush=True)
        return train_loss

    def evaluate(self, loader: DataLoader) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer is not attached to an inner trainer.")
        return self.inner.evaluate(loader)

    def fit(
        self,
        data: Any,
        val_data: Optional[Any] = None,
        epochs: int = 1,
        max_tokens: int = 4000,
        batch_size: int = 16,
    ) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer is not attached to an inner trainer.")
        pin = self.inner.device.type == "cuda"
        if hasattr(data, "load_samples"):
            train_samples, val_samples = data.load_samples()
            train_loader = self.inner.collator.pack_loader(train_samples, max_tokens=max_tokens, batch_size=batch_size, pin_memory=pin)
            self.val_loader = self.inner.collator.pack_loader(val_samples, max_tokens=max_tokens, batch_size=batch_size, shuffle=False, pin_memory=pin)
        elif isinstance(data, (list, tuple)):
            train_loader = self.inner.collator.pack_loader(data, max_tokens=max_tokens, batch_size=batch_size, pin_memory=pin)
            self.val_loader = self.inner.collator.pack_loader(val_data, max_tokens=max_tokens, batch_size=batch_size, shuffle=False, pin_memory=pin) if val_data else None
        else:
            train_loader = data
            if val_data:
                self.val_loader = val_data

        last_loss = 0.0
        for epoch in range(1, epochs + 1):
            last_loss = self.train_epoch(train_loader)
            val_info = f" | Best Val Loss: {self.best_val_loss:.4f}" if self.val_loader else ""
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {last_loss:.4f}{val_info}")
        return last_loss

__all__ = ["CheckpointingLayer"]
