from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Union
from training.checkpoints import CheckpointMeta
from training.batches import StepStats

if TYPE_CHECKING:
    from training.epoch_trainer import ITrainer

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

    def save(self, meta: Optional[CheckpointMeta] = None) -> None:
        if self.inner is not None:
            self.inner.save(meta=meta)

    def fit(self, train_data, epochs: int = 1, **kwargs) -> Iterator[StepStats]:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        last_log = time.perf_counter()
        for s in self.inner.fit(train_data, epochs=epochs, **kwargs):
            now = time.perf_counter()
            if s.step == s.total_steps:
                val_info = f" | Val Loss: {s.val_loss:.4f}" if s.val_loss is not None else ""
                self.sink(f"Epoch {s.epoch}/{s.total_epochs} Complete | Train Loss: {s.loss:.4f}{val_info}")
                last_log = now
            elif now - last_log >= self.log_interval_sec:
                self.sink(f"Epoch {s.epoch}/{s.total_epochs} | Step {s.step}/{s.total_steps} | Loss: {s.loss:.4f}")
                last_log = now
            yield s

    def __or__(self, layer: Union[type, Callable[["ITrainer"], "ITrainer"]]) -> "ITrainer":
        return layer(self) if isinstance(layer, type) else (getattr(layer, "attach")(self) if hasattr(layer, "attach") else layer(self))

__all__ = ["MetricsLayer"]
