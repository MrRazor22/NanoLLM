from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Union
from pipeline import PipelineComposable
from nanollm.training.epoch_trainer import Batch, EpochStats, ITrainer, StepStats

class MetricsLayer(PipelineComposable, ITrainer):
    """ATA Composable Layer: logs step + epoch stats, persists metrics to disk."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        output_path: Optional[Union[str, Path]] = None,
        sink: Optional[Callable[[str], None]] = print,
        prefix: str = "",
        log_every_n_steps: int = 50,
    ):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.sink = sink
        self.prefix = prefix
        self.log_every_n_steps = log_every_n_steps
        self.history: List[EpochStats] = []

    def attach(self, inner: ITrainer) -> "MetricsLayer":
        self.inner = inner
        return self

    @property
    def model(self) -> Any:
        return getattr(self.inner, "model")

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
        pfx = f"[{self.prefix}] " if self.prefix else ""
        for stats in self.inner.fit(train_batches, val_batches=val_batches, epochs=epochs):
            if self.sink:
                total = stats.step_stats[-1].total_steps if stats.step_stats else 0
                total_str = f"/{total}" if total else ""
                for s in stats.step_stats:
                    if s.step % self.log_every_n_steps == 0 or s.step == total:
                        self.sink(f"{pfx}  Step [{s.step:5d}{total_str}] Loss: {s.loss:.4f} | {s.elapsed_ms:.0f}ms/step")
                val_info = f" | Val Loss: {stats.val_loss:.4f}" if stats.val_loss is not None else ""
                self.sink(f"{pfx}Epoch {stats.epoch:2d}/{stats.total_epochs:2d} | Train Loss: {stats.train_loss:.4f}{val_info} | Elapsed: {stats.elapsed_sec:.1f}s")
            self.history.append(stats)
            if self.output_path:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_path, "w", encoding="utf-8") as f:
                    json.dump([{**asdict(s), "step_stats": []} for s in self.history], f, indent=2)
            yield stats

__all__ = ["MetricsLayer"]
