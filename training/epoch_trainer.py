import time
from typing import Iterator, Optional, Protocol, Sequence, runtime_checkable
import torch

from data.case import DecisionCase
from model.torch_model import IModel, TorchModel
from training.policies.batch import BucketedBatch, IBatch, SequentialBatch
from training.policies.loss import CalibratedLoss, ILoss
from training.policies.schedule import CosineSchedule, ISchedule
from training.stats import StepStats

@runtime_checkable
class ITrainerLayer(Protocol):
    def __call__(self, inner: "ITrainer") -> "ITrainer": ...

@runtime_checkable
class ITrainer(Protocol):
    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]: ...

    def evaluate(self, cases: Sequence[DecisionCase]) -> float: ...
    def __or__(self, layer: ITrainerLayer) -> "ITrainer": ...

class EpochTrainer(ITrainer):
    def __init__(
        self,
        model: Optional[IModel] = None,
        batch_size: int = 16,
        base_lr: float = 2e-5,
        accum_steps: int = 1,
        schedule: Optional[ISchedule] = None,
        batching: Optional[IBatch] = None,
        eval_batching: Optional[IBatch] = None,
        loss: Optional[ILoss] = None,
        warmup_ratio: float = 0.05,
    ):
        self.model = model or TorchModel(accum_steps=accum_steps, lr=base_lr)
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.accum_steps = max(1, accum_steps)
        self.schedule = schedule or CosineSchedule(warmup_ratio=warmup_ratio)
        self.batching = batching or BucketedBatch()
        self.eval_batching = eval_batching or SequentialBatch()
        self.loss = loss or CalibratedLoss()

    def evaluate(self, cases: Sequence[DecisionCase]) -> float:
        batches = list(self.eval_batching.batch(cases, batch_size=self.batch_size))
        if not batches:
            return 0.0
        total_loss = 0.0
        with torch.no_grad():
            for batch in batches:
                scores = self.model.forward(batch, train=False)
                targets = torch.as_tensor([c.target_index() for c in batch], device=scores.device)
                loss_tensor = self.loss.compute(scores, targets)
                total_loss += loss_tensor.item()
        return total_loss / len(batches)

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
                scores = self.model.forward(batch, train=True)
                targets = torch.as_tensor([c.target_index() for c in batch], device=scores.device)
                loss_tensor = self.loss.compute(scores, targets)

                should_commit = ((n + 1) % self.accum_steps == 0) or (n + 1 == total_batches)
                self.model.step(loss_tensor, lr=cur_lr, commit=should_commit)
                dt = (time.perf_counter() - t0) * 1000.0
                total_loss += loss_tensor.item()

                is_val_cadence = val_cases and (n + 1 == total_batches or n + 1 == (total_batches // 2))
                val_loss = self.evaluate(val_cases) if is_val_cadence else None

                yield StepStats(
                    epoch=ep,
                    total_epochs=epochs,
                    step=step,
                    total_steps=total_steps,
                    loss=total_loss / (n + 1),
                    elapsed_ms=dt,
                    lr=cur_lr,
                    val_loss=val_loss,
                    batch=n + 1,
                    total_batches=total_batches,
                )

    def __or__(self, layer: ITrainerLayer) -> "ITrainer":
        return layer(self)
