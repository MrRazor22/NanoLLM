from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Optional, Union
from inference.questions import Composable
from training.batches import StepStats, TrainingBatch

if TYPE_CHECKING:
    from training.epoch_trainer import ITrainer

class MetricsLayer(Composable):
    """Layer Decorator (lambda: P -> P): Real-time logging over the StepStats stream."""
    def __init__(self, inner: Optional["ITrainer"] = None, output_path: Optional[Union[str, Path]] = None, sink: Optional[Callable[[str], None]] = None, log_interval_sec: float = 10.0):
        self.inner, self.output_path, self.sink, self.log_interval_sec = inner, Path(output_path) if output_path else None, sink or (lambda msg: print(msg, flush=True)), log_interval_sec

    def attach(self, inner: "ITrainer") -> "MetricsLayer":
        self.inner = inner
        return self

    def evaluate(self, data, **kwargs) -> float:
        if self.inner is None: raise RuntimeError("MetricsLayer has no inner trainer.")
        return self.inner.evaluate(data, **kwargs)

    def fit(self, train_data, epochs: int = 1, **kwargs) -> Iterator[StepStats]:
        if self.inner is None: raise RuntimeError("MetricsLayer has no inner trainer.")
        last_log = time.perf_counter()
        for s in self.inner.fit(train_data, epochs=epochs, **kwargs):
            now = time.perf_counter()
            if now - last_log >= self.log_interval_sec:
                self.sink(f"Epoch {s.epoch}/{s.total_epochs} | Step {s.step}/{s.total_steps} | Loss: {s.loss:.4f}")
                last_log = now
            yield s

__all__ = ["MetricsLayer"]
