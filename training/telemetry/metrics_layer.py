import time
from typing import Iterator, Optional, Protocol, Sequence, runtime_checkable
from rich.console import Console
from data.case import DecisionCase
from model.device import resolve_vram_gb
from training.stats import StepStats
from training.epoch_trainer import ITrainer, ITrainerLayer

@runtime_checkable
class ILogSink(Protocol):
    """Opinion Sub-Primitive: Stream sink for metric log messages."""
    def write(self, message: str) -> None: ...

class ConsoleSink(ILogSink):
    """Rich stdout logging sink with markup support."""
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def write(self, message: str) -> None:
        self.console.print(message)

class MetricsLayer(ITrainer):
    """Layer Decorator (lambda: P -> P): Real-time logging over the StepStats stream."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        sink: Optional[ILogSink] = None,
        log_interval_sec: float = 10.0,
    ):
        self.inner = inner
        self.sink = sink or ConsoleSink()
        self.log_interval_sec = log_interval_sec

    @property
    def model(self):
        return getattr(self.inner, "model", None)

    def __call__(self, inner: ITrainer) -> "MetricsLayer":
        self.inner = inner
        return self

    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        start_time = time.perf_counter()
        last_log = start_time
        for s in self.inner.fit(cases, epochs=epochs, val_cases=val_cases):
            now = time.perf_counter()
            is_epoch_end = (s.batch == s.total_batches)
            is_checkpoint = (s.val_loss is not None)
            is_interval = (now - last_log >= self.log_interval_sec)

            if is_epoch_end or is_checkpoint or is_interval:
                el_s = int(now - start_time)
                el_str = f"{el_s // 60:02d}:{el_s % 60:02d}"
                rem_s = int((el_s / max(1, s.step)) * max(0, s.total_steps - s.step))
                eta_str = f"{rem_s // 60:02d}:{rem_s % 60:02d}"
                _v = resolve_vram_gb()
                vram = f" | VRAM: {_v:.1f}GB" if _v is not None else ""
                val = f" | [bold green]Val Loss: {s.val_loss:.4f}[/bold green]" if s.val_loss is not None else ""

                if is_epoch_end:
                    msg = f"[bold cyan]Epoch {s.epoch}/{s.total_epochs} Complete[/bold cyan] | [green]Train Loss: {s.loss:.4f}[/green]{val}{vram} | [dim]{s.elapsed_ms:.0f}ms/step[/dim] | Total: {el_str}"
                elif is_checkpoint:
                    pct = int((s.batch / max(1, s.total_batches)) * 100)
                    msg = f"[cyan]Epoch {s.epoch}/{s.total_epochs}[/cyan] Checkpoint ({pct}%) | [green]Train: {s.loss:.4f}[/green]{val}{vram} | [dim]Elapsed: {el_str} | ETA: {eta_str}[/dim]"
                else:
                    pct = int((s.step / max(1, s.total_steps)) * 100)
                    lr = f" | [cyan]LR: {s.lr:.2e}[/cyan]" if s.lr > 0 else ""
                    msg = f"[cyan]Epoch {s.epoch}/{s.total_epochs}[/cyan] | Step [bold]{s.step}[/bold]/{s.total_steps} ({pct}%) | [green]Loss: {s.loss:.4f}[/green]{lr}{vram} | [dim]{s.elapsed_ms:.0f}ms/step[/dim] | [dim]Elapsed: {el_str} | ETA: {eta_str}[/dim]"

                self.sink.write(msg)
                last_log = now

            yield s

    def evaluate(self, cases: Sequence[DecisionCase]) -> float:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        return self.inner.evaluate(cases)

    def __or__(self, layer: ITrainerLayer) -> ITrainer:
        return layer(self)
