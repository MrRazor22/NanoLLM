from pathlib import Path
import time
from typing import Callable, Iterator, Optional, Union
import torch

from model.layouts.checkpoints import CheckpointMeta
from training.stats import StepStats
from training.epoch_trainer import ITrainer

class MetricsLayer(ITrainer):
    """Layer Decorator (lambda: P -> P): Real-time logging over the StepStats stream."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        output_path: Optional[Union[str, Path]] = None,
        sink: Optional[Callable[[str], None]] = None,
        log_interval_sec: float = 10.0,
    ):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.sink = sink or (lambda msg: print(msg, flush=True))
        self.log_interval_sec = log_interval_sec

    def __call__(self, inner: ITrainer) -> "MetricsLayer":
        self.inner = inner
        return self

    attach = __call__

    def save(self, meta: Optional[CheckpointMeta] = None) -> None:
        if self.inner is not None:
            self.inner.save(meta=meta)

    def fit(self, train_data, epochs: int = 1, **kwargs) -> Iterator[StepStats]:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        start_time = time.perf_counter()
        last_log = start_time
        for s in self.inner.fit(train_data, epochs=epochs, **kwargs):
            now = time.perf_counter()
            elapsed_sec = int(now - start_time)
            elapsed_str = f"{elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}"
            rem_steps = max(0, s.total_steps - s.step)
            rem_sec = int((elapsed_sec / max(1, s.step)) * rem_steps)
            eta_str = f"{rem_sec // 60:02d}:{rem_sec % 60:02d}"

            vram_info = f" | VRAM: {torch.cuda.memory_allocated() / (1024**3):.1f}GB" if torch.cuda.is_available() else ""

            if s.step == s.total_steps:
                val_info = f" | Val Loss: {s.val_loss:.4f}" if s.val_loss is not None else ""
                self.sink(f"Epoch {s.epoch}/{s.total_epochs} Complete | Train Loss: {s.loss:.4f}{val_info}{vram_info} | {s.elapsed_ms:.0f}ms/step | Total: {elapsed_str}")
                last_log = now
            elif s.val_loss is not None:
                pct = int((s.step / max(1, s.total_steps)) * 100)
                self.sink(f"Epoch {s.epoch}/{s.total_epochs} Checkpoint ({pct}%) | Train Loss: {s.loss:.4f} | Val Loss: {s.val_loss:.4f}{vram_info} | Elapsed: {elapsed_str} | ETA: {eta_str}")
                last_log = now
            elif now - last_log >= self.log_interval_sec:
                pct = int((s.step / max(1, s.total_steps)) * 100)
                self.sink(f"Epoch {s.epoch}/{s.total_epochs} | Step {s.step}/{s.total_steps} ({pct}%) | Loss: {s.loss:.4f}{vram_info} | {s.elapsed_ms:.0f}ms/step | Elapsed: {elapsed_str} | ETA: {eta_str}")
                last_log = now
            yield s

    def __or__(self, layer: Callable[[ITrainer], ITrainer]) -> ITrainer:
        return layer(self)
