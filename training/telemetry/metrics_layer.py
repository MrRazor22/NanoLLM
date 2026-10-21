import time
from typing import Iterator, Optional, Sequence
from rich.console import Console

from data.case import DecisionCase
from training.epoch_trainer import ITrainer, ITrainerLayer
from training.stats import StepStats

console = Console()

class MetricsLayer(ITrainer):
    """Layer Decorator (lambda: P -> P): Sovereign terminal and progress telemetry."""

    def __init__(self, inner: Optional[ITrainer] = None, log_interval_sec: float = 2.0):
        self.inner = inner
        self.log_interval_sec = log_interval_sec
        self.last_log_time = 0.0

    def __call__(self, inner: ITrainer) -> "MetricsLayer":
        self.inner = inner
        return self

    def fit(
        self,
        train_cases: Sequence[DecisionCase],
        val_cases: Sequence[DecisionCase],
        epochs: int = 1,
    ) -> Iterator[StepStats]:
        if not self.inner:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        t_epoch_start = time.perf_counter()
        for s in self.inner.fit(train_cases, val_cases, epochs=epochs):
            now = time.perf_counter()
            is_cadence = (now - self.last_log_time >= self.log_interval_sec) or (s.batch == s.total_batches)
            if is_cadence:
                self.last_log_time = now
                val_text = f" | val_loss: [bold red]{s.val_loss:.4f}[/]" if s.val_loss is not None else ""
                console.print(
                    f"Epoch {s.epoch}/{s.total_epochs} | "
                    f"Batch {s.batch}/{s.total_batches} | "
                    f"Loss: [bold green]{s.loss:.4f}[/] | "
                    f"LR: {s.lr:.2e} | "
                    f"Step: {s.elapsed_ms:.1f}ms{val_text}"
                )
            if s.batch == s.total_batches:
                console.print(f"[bold cyan]Epoch {s.epoch} complete[/] (took {time.perf_counter() - t_epoch_start:.1f}s)")
                t_epoch_start = time.perf_counter()
            yield s

    def evaluate(self, cases: Sequence[DecisionCase]) -> float:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        return self.inner.evaluate(cases)

    def __or__(self, layer: ITrainerLayer) -> ITrainer:
        return layer(self)
