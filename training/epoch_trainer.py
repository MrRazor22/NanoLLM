import time
from typing import Callable, Iterator, Optional, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase
from model.torch_model import IModel, TorchModel
from training.policies.batch import BucketedBatch, IBatch
from training.policies.schedule import CosineSchedule, ISchedule
from training.stats import EvalStats, StepStats

@runtime_checkable
class ITrainer(Protocol):
    """Core Primitive: Operational boundary for model optimization and evaluation."""
    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]: ...

    def evaluate(
        self,
        cases: Sequence[DecisionCase],
        on_progress: Optional[Callable[[EvalStats], None]] = None,
    ) -> float: ...

class EpochTrainer(ITrainer):
    """Core Primitive Implementation: Supervised training loop over domain decision cases."""

    def __init__(
        self,
        model: Optional[IModel] = None,
        batch_size: int = 16,
        base_lr: float = 2e-5,
        schedule: Optional[ISchedule] = None,
        batching: Optional[IBatch] = None,
    ):
        self.model = model or TorchModel(accum_steps=1)
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.schedule = schedule or CosineSchedule()
        self.batching = batching or BucketedBatch()

    def evaluate(
        self,
        cases: Sequence[DecisionCase],
        on_progress: Optional[Callable[[EvalStats], None]] = None,
    ) -> float:
        """Evaluates model performance over candidate cases with streaming progress telemetry."""
        import torch
        batches = list(self.batching.batch(cases, batch_size=self.batch_size))
        total = len(batches)
        total_loss = 0.0
        start = time.perf_counter()
        last_log = start
        for i, b in enumerate(batches, 1):
            loss = self.model.eval_step(b)
            total_loss += loss
            now = time.perf_counter()
            if on_progress and (now - last_log >= 3.0 or i == total):
                vram = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
                on_progress(EvalStats(step=i, total_steps=total, loss=total_loss / i, elapsed_sec=now - start, vram_gb=vram))
                last_log = now
        return total_loss / max(1, total)

    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]:
        items = list(cases)
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size if items else 0
        total_steps = epochs * total_batches

        for ep in range(1, epochs + 1):
            total_loss = 0.0
            batches = list(self.batching.batch(items, batch_size=self.batch_size))

            for n, batch in enumerate(batches):
                step = (ep - 1) * total_batches + (n + 1)
                cur_lr = self.schedule.get_lr(step, total_steps, self.base_lr)

                t0 = time.perf_counter()
                loss = self.model.train_step(batch, lr=cur_lr)
                dt = (time.perf_counter() - t0) * 1000.0
                total_loss += loss

                is_val_cadence = val_cases and (n + 1 == total_batches or n + 1 == (total_batches // 2))
                val_loss = self.evaluate(val_cases) if is_val_cadence else None

                yield StepStats(
                    epoch=ep,
                    total_epochs=epochs,
                    step=n + 1,
                    total_steps=total_batches,
                    loss=total_loss / (n + 1),
                    elapsed_ms=dt,
                    lr=cur_lr,
                    val_loss=val_loss,
                    batch=n + 1,
                    total_batches=total_batches,
                )

    def __or__(self, layer: Callable[["ITrainer"], "ITrainer"]) -> "ITrainer":
        return layer(self)
