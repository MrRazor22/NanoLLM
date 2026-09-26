from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Union
from torch.utils.data import DataLoader
from pipeline import PipelineComposable
from nanollm.training.epoch_trainer import EpochStats, ITrainer

class MetricsLayer(PipelineComposable, ITrainer):
    """ATA Composable Layer: captures training history, persists metrics to disk, and routes telemetry to a pluggable sink."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        output_path: Optional[Union[str, Path]] = None,
        sink: Optional[Callable[[str], None]] = print,
        prefix: str = "",
    ):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.sink = sink
        self.prefix = prefix
        self.history: List[EpochStats] = []

    def attach(self, inner: ITrainer) -> "MetricsLayer":
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

    def fit_iter(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ):
        if self.inner is None:
            raise RuntimeError("MetricsLayer is not attached to an inner trainer.")
        for stats in self.inner.fit_iter(train_loader, val_loader=val_loader, epochs=epochs):
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

__all__ = ["MetricsLayer"]
