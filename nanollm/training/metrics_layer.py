from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Union
from pipeline import PipelineComposable
from nanollm.training.epoch_trainer import Batch, EpochStats, ITrainer

class MetricsLayer(PipelineComposable, ITrainer):
    """ATA Composable Layer: captures training history, persists metrics to disk, and routes telemetry to a pluggable sink."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        output_path: Optional[Union[str, Path]] = None,
        sink: Optional[Callable[[str], None]] = print,
        prefix: str = "",
    ):
        if inner is not None and not isinstance(inner, ITrainer):
            raise TypeError(f"inner must implement ITrainer, got {type(inner).__name__}")
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.sink = sink
        self.prefix = prefix
        self.history: List[EpochStats] = []

    def attach(self, inner: ITrainer) -> "MetricsLayer":
        if not isinstance(inner, ITrainer):
            raise TypeError(f"inner must implement ITrainer, got {type(inner).__name__}")
        self.inner = inner
        return self

    @property
    def model(self) -> Any:
        return getattr(self.inner, "model")

    def _persist(self) -> None:
        if not self.output_path:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(s) for s in self.history]
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def train_epoch(self, batches: Iterable[Batch]) -> float:
        if self.inner is None:
            raise RuntimeError("MetricsLayer is not attached to an inner trainer.")
        return self.inner.train_epoch(batches)

    def evaluate(self, batches: Iterable[Batch]) -> float:
        if self.inner is None:
            raise RuntimeError("MetricsLayer is not attached to an inner trainer.")
        return self.inner.evaluate(batches)

    def fit(
        self,
        train_batches: Iterable[Batch],
        val_batches: Optional[Iterable[Batch]] = None,
        epochs: int = 1,
    ) -> Iterator[EpochStats]:
        if self.inner is None:
            raise RuntimeError("MetricsLayer is not attached to an inner trainer.")
        for stats in self.inner.fit(train_batches, val_batches=val_batches, epochs=epochs):
            self.history.append(stats)
            if self.sink:
                pfx = f"[{self.prefix}] " if self.prefix else ""
                val_info = f" | Val Loss: {stats.val_loss:.4f}" if stats.val_loss is not None else ""
                self.sink(
                    f"{pfx}Epoch {stats.epoch:2d}/{stats.total_epochs:2d} | "
                    f"Train Loss: {stats.train_loss:.4f}{val_info} | "
                    f"Elapsed: {stats.elapsed_sec:.1f}s"
                )
            self._persist()
            yield stats

__all__ = ["MetricsLayer"]
