import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Union
import torch
from pipeline import PipelineComposable
from nanollm.training.epoch_trainer import Batch, ITrainer, StepStats

class MetricsLayer(PipelineComposable, ITrainer):
    """ATA Layer: time-based logging over the StepStats stream from the primitive."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        output_path: Optional[Union[str, Path]] = None,
        sink: Optional[Callable[[str], None]] = None,
        log_interval_sec: float = 10.0,
    ):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.sink = sink if sink is not None else (lambda msg: print(msg, flush=True))
        self.log_interval_sec = log_interval_sec
        self.history: List[StepStats] = []

    def attach(self, inner: ITrainer) -> "MetricsLayer":
        self.inner = inner
        return self

    @property
    def model(self) -> Any:
        return getattr(self.inner, "model")

    def evaluate(self, batches: Iterable[Batch]) -> float:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        return self.inner.evaluate(batches)

    def fit(
        self,
        train_batches: Iterable[Batch],
        val_batches: Optional[Iterable[Batch]] = None,
        epochs: int = 1,
    ) -> Iterator[StepStats]:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        seen_epochs: set = set()
        last_log_t = time.perf_counter()
        for s in self.inner.fit(train_batches, val_batches=val_batches, epochs=epochs):
            self.history.append(s)
            if self.sink:
                total_str = f"/{s.total_steps}" if s.total_steps else ""
                if s.epoch not in seen_epochs:
                    seen_epochs.add(s.epoch)
                    self.sink(f"Epoch {s.epoch}/{s.total_epochs} - first step done, training live.")
                    last_log_t = time.perf_counter()
                now = time.perf_counter()
                if now - last_log_t >= self.log_interval_sec or s.val_loss is not None:
                    remaining = s.total_steps - s.step if s.total_steps else 0
                    if remaining > 0 and s.elapsed_ms > 0:
                        total_sec = int((remaining * s.elapsed_ms) / 1000.0)
                        mins, secs = divmod(total_sec, 60)
                        eta_str = f" | ETA: {mins}m {secs:02d}s" if mins > 0 else f" | ETA: {secs}s"
                    else:
                        eta_str = ""
                    gpu_str = ""
                    if torch.cuda.is_available():
                        vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                        gpu_str = f" | VRAM: {vram_gb:.2f}GB"
                    lr_str = f" | LR: {s.lr:.1e}" if s.lr > 0 else ""
                    batch_str = f" (Batch {s.batch}/{s.total_batches})" if s.total_batches > 0 else ""
                    line = f"  Ep {s.epoch}/{s.total_epochs} | Step [{s.step:5d}{total_str}]{batch_str} Loss: {s.loss:.4f} | Speed: {s.elapsed_ms:.1f}ms/step{eta_str}{lr_str}{gpu_str}"
                    if s.val_loss is not None:
                        line += f" | Val Loss: {s.val_loss:.4f}"
                    self.sink(line)
                    last_log_t = now
            if self.output_path and s.val_loss is not None:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_path.write_text(json.dumps(
                    [{"epoch": x.epoch, "train_loss": x.loss, "val_loss": x.val_loss}
                     for x in self.history if x.val_loss is not None], indent=2))
            yield s

__all__ = ["MetricsLayer"]
