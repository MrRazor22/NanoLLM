from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
import torch
from pipeline import PipelineComposable
from nanollm.training.epoch_trainer import Batch, EpochStats, ITrainer

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

    def train_epoch(self, batches: Iterable[Batch]) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer is not attached to an inner trainer.")
        return self.inner.train_epoch(batches)

    def evaluate(self, batches: Iterable[Batch]) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer is not attached to an inner trainer.")
        return self.inner.evaluate(batches)

    def fit(
        self,
        train_batches: Iterable[Batch],
        val_batches: Optional[Iterable[Batch]] = None,
        epochs: int = 1,
    ) -> Iterator[EpochStats]:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer is not attached to an inner trainer.")
        for stats in self.inner.fit(train_batches, val_batches=val_batches, epochs=epochs):
            if stats.val_loss is not None:
                if stats.val_loss < self.best_val_loss:
                    self.best_val_loss = stats.val_loss
                    if self.output_path:
                        self.save_checkpoint(self.output_path)
            elif self.output_path:
                self.save_checkpoint(self.output_path)
            yield stats

__all__ = ["CheckpointingLayer"]

