from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Optional, Union
from nanollm.trainer.batches import StepStats, TrainingBatch

if TYPE_CHECKING:
    from nanollm.trainer.layers.checkpointing_layer import CheckpointingLayer
    from nanollm.trainer.epoch_trainer import ITrainer

class MetricsLayer:
    """Layer Decorator (lambda: P -> P): Real-time logging over the StepStats stream."""

    def __init__(
        self,
        inner: Optional["ITrainer"] = None,
        output_path: Optional[Union[str, Path]] = None,
        sink: Optional[Callable[[str], None]] = None,
        log_interval_sec: float = 10.0,
    ):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.sink = sink or (lambda msg: print(msg, flush=True))
        self.log_interval_sec = log_interval_sec

    def attach(self, inner: "ITrainer") -> "MetricsLayer":
        self.inner = inner
        return self

    def evaluate(self, batches: Iterable[TrainingBatch]) -> float:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        return self.inner.evaluate(batches)

    def fit(self, train_batches: Iterable[TrainingBatch], val_batches: Optional[Iterable[TrainingBatch]] = None, epochs: int = 1) -> Iterator[StepStats]:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        last_log = time.perf_counter()
        for s in self.inner.fit(train_batches, val_batches=val_batches, epochs=epochs):
            now = time.perf_counter()
            if now - last_log >= self.log_interval_sec or s.val_loss is not None:
                val_str = f" | Val: {s.val_loss:.4f}" if s.val_loss is not None else ""
                self.sink(f"Epoch {s.epoch}/{s.total_epochs} | Step {s.step}/{s.total_steps} | Loss: {s.loss:.4f}{val_str}")
                last_log = now
            yield s

    def add(self, layer: Union[type, "CheckpointingLayer", "MetricsLayer"]) -> "ITrainer":
        if isinstance(layer, type):
            return layer(self)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer

    def __or__(self, layer: Union[type, "CheckpointingLayer", "MetricsLayer"]) -> "ITrainer":
        return self.add(layer)

__all__ = ["MetricsLayer"]
