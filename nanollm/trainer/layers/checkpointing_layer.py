from pathlib import Path
import time
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Union
import torch
import torch.nn as nn
from nanollm.trainer.policies.registry import get_next_checkpoint_path
from nanollm.trainer.batches import StepStats, TrainingBatch

if TYPE_CHECKING:
    from nanollm.trainer.epoch_trainer import ITrainer

class CheckpointingLayer:
    """Layer Decorator (lambda: P -> P): Saves model checkpoint at epoch-end."""

    def __init__(self, inner: Optional["ITrainer"] = None, output_path: Optional[str] = None):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.best_val_loss = float("inf")

    def attach(self, inner: "ITrainer") -> "CheckpointingLayer":
        self.inner = inner
        return self

    @property
    def model(self) -> nn.Module:
        return getattr(self.inner, "model")

    def _save(self, path: Path, stats: Optional[StepStats] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": path.name,
            "epoch": stats.epoch if stats else 1,
            "train_loss": round(stats.loss, 6) if stats else None,
            "val_loss": round(stats.val_loss, 6) if stats and stats.val_loss is not None else None,
        }
        torch.save({"model_state_dict": self.model.state_dict(), "meta": meta}, str(path))

    def evaluate(self, batches: Iterable[TrainingBatch]) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        return self.inner.evaluate(batches)

    def fit(self, train_batches: Iterable[TrainingBatch], val_batches: Optional[Iterable[TrainingBatch]] = None, epochs: int = 1) -> Iterator[StepStats]:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        for s in self.inner.fit(train_batches, val_batches=val_batches, epochs=epochs):
            if s.val_loss is not None and s.val_loss < self.best_val_loss:
                self.best_val_loss = s.val_loss
                save_dest = self.output_path or get_next_checkpoint_path()
                self._save(save_dest, stats=s)
            yield s

    def add(self, layer: Union[type, "CheckpointingLayer", "MetricsLayer"]) -> "ITrainer":
        if isinstance(layer, type):
            return layer(self)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer

    def __or__(self, layer: Union[type, "CheckpointingLayer", "MetricsLayer"]) -> "ITrainer":
        return self.add(layer)

from nanollm.trainer.layers.metrics_layer import MetricsLayer

__all__ = ["CheckpointingLayer"]
