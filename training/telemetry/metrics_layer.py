from pathlib import Path
import time
from typing import Iterator, Optional, Protocol, Union, runtime_checkable
import torch
from training.stats import StepStats
from training.epoch_trainer import ITrainer, ITrainerLayer

@runtime_checkable
class ILogSink(Protocol):
    """Opinion Sub-Primitive: Stream sink for metric log messages."""
    def write(self, message: str) -> None: ...

class ConsoleSink(ILogSink):
    """Default stdout logging sink."""
    def write(self, message: str) -> None:
        print(message, flush=True)

class MetricsLayer(ITrainer):
    """Layer Decorator (lambda: P -> P): Real-time logging over the StepStats stream."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        output_path: Optional[Union[str, Path]] = None,
        sink: Optional[ILogSink] = None,
        log_interval_sec: float = 10.0,
    ):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.sink = sink or ConsoleSink()
        self.log_interval_sec = log_interval_sec

    @property
    def model(self):
        return getattr(self.inner, "model", None)

    def __call__(self, inner: ITrainer) -> "MetricsLayer":
        self.inner = inner
        return self

    def fit(self, train_data, epochs: int = 1, **kwargs) -> Iterator[StepStats]:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        start_time = time.perf_counter()
        last_log = start_time
        for s in self.inner.fit(train_data, epochs=epochs, **kwargs):
            now = time.perf_counter()
            is_epoch_end = (s.batch == s.total_batches)
            is_checkpoint = (s.val_loss is not None)
            is_interval = (now - last_log >= self.log_interval_sec)

            if is_epoch_end or is_checkpoint or is_interval:
                elapsed_sec = int(now - start_time)
                elapsed_str = f"{elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}"
                rem_steps = max(0, s.total_steps - s.step)
                rem_sec = int((elapsed_sec / max(1, s.step)) * rem_steps)
                eta_str = f"{rem_sec // 60:02d}:{rem_sec % 60:02d}"
                vram_info = f" | VRAM: {torch.cuda.memory_reserved() / (1024**3):.1f}GB" if torch.cuda.is_available() else ""
                pct = int((s.step / max(1, s.total_steps)) * 100)
                batch_pct = int((s.batch / max(1, s.total_batches)) * 100)

                if is_epoch_end:
                    val_info = f" | Val Loss: {s.val_loss:.4f}" if s.val_loss is not None else ""
                    self.sink.write(f"Epoch {s.epoch}/{s.total_epochs} Complete | Train Loss: {s.loss:.4f}{val_info}{vram_info} | {s.elapsed_ms:.0f}ms/step | Total: {elapsed_str}")
                elif is_checkpoint:
                    self.sink.write(f"Epoch {s.epoch}/{s.total_epochs} Checkpoint ({batch_pct}%) | Train Loss: {s.loss:.4f} | Val Loss: {s.val_loss:.4f}{vram_info} | Elapsed: {elapsed_str} | ETA: {eta_str}")
                else:
                    lr_info = f" | LR: {s.lr:.2e}" if s.lr > 0 else ""
                    self.sink.write(f"Epoch {s.epoch}/{s.total_epochs} | Step {s.step}/{s.total_steps} ({pct}%) | Loss: {s.loss:.4f}{lr_info}{vram_info} | {s.elapsed_ms:.0f}ms/step | Elapsed: {elapsed_str} | ETA: {eta_str}")

                last_log = now

            yield s

    def evaluate(self, cases) -> float:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        return self.inner.evaluate(cases)

    def __or__(self, layer: ITrainerLayer) -> ITrainer:
        return layer(self)
