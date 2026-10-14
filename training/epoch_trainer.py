import time
from typing import Callable, Iterator, Optional, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase
from model.torch_model import IModel, TorchModel
from training.policies.batch import BucketedBatch, IBatch
from training.policies.schedule import CosineSchedule, ISchedule
from training.stats import StepStats

@runtime_checkable
class ITrainer(Protocol):
    """Core Primitive: Operational boundary for model optimization."""
    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]: ...

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

    def evaluate(self, cases: Sequence[DecisionCase]) -> float:
        """Evaluates model performance over candidate cases using batch policy."""
        batches = list(self.batching.batch(cases, batch_size=self.batch_size))
        return sum(self.model.eval_step(b) for b in batches) / max(1, len(batches))

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
