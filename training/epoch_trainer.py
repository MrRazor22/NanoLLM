from dataclasses import dataclass
import itertools
import logging
from pathlib import Path
import random
import time
from typing import Iterator, Optional, Protocol, Sequence, runtime_checkable
import torch

from data.case import DecisionCase
from model.torch_model import IModel, TorchModel
from model.weights import WeightInfo
from training.policies.learner import AdamWLearner, ILearner
from training.policies.loss import CalibratedLoss, ILoss
from training.policies.schedule import CosineSchedule, ISchedule
from training.policies.transform import ICaseTransform, ShuffleCandidates

logger = logging.getLogger("piqo.training")

@dataclass(frozen=True, slots=True)
class TrainingStats(WeightInfo):
    epoch: int = 1
    total_epochs: int = 1
    step: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    lr: float = 0.0
    elapsed_ms: float = 0.0
    batch: int = 0
    total_batches: int = 0

@runtime_checkable
class ITrainerLayer(Protocol):
    def __call__(self, inner: "ITrainer") -> "ITrainer": ...

@runtime_checkable
class ITrainer(Protocol):
    def train(
        self,
        train_cases: Sequence[DecisionCase],
        val_cases: Sequence[DecisionCase],
        epochs: int = 1,
    ) -> Iterator[TrainingStats]: ...

    def evaluate(self, cases: Sequence[DecisionCase]) -> float: ...
    def __or__(self, layer: ITrainerLayer) -> "ITrainer": ...

class EpochTrainer(ITrainer):
    def __init__(
        self,
        model: Optional[TorchModel] = None,
        batch_size: int = 16,
        learner: Optional[ILearner] = None,
        schedule: Optional[ISchedule] = None,
        loss: Optional[ILoss] = None,
        case_transform: Optional[ICaseTransform] = None,
    ):
        self.model = model or TorchModel()
        self.batch_size = batch_size
        self.learner = learner or AdamWLearner(self.model.parameters())
        self.schedule = schedule or CosineSchedule()
        self.loss = loss or CalibratedLoss()
        self.case_transform = case_transform or ShuffleCandidates()

    def evaluate(self, cases: Sequence[DecisionCase]) -> float:
        batches = list(itertools.batched(cases, self.batch_size))
        if not batches:
            return 0.0
        total_loss = 0.0
        with torch.no_grad():
            for batch in batches:
                scores = self.model.score(batch, train=False)
                targets = torch.as_tensor([c.target_index() for c in batch], device=scores.device)
                total_loss += self.loss.compute(scores, targets).item()
        return total_loss / len(batches)

    def train(
        self,
        train_cases: Sequence[DecisionCase],
        val_cases: Sequence[DecisionCase],
        epochs: int = 1,
    ) -> Iterator[TrainingStats]:
        items = list(train_cases)
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size if self.batch_size else 0
        total_steps = epochs * total_batches
        step = 0

        for ep in range(1, epochs + 1):
            total_loss = 0.0
            epoch_items = list(items)
            random.shuffle(epoch_items)
            batches = list(itertools.batched(epoch_items, self.batch_size))

            for batch_index, batch in enumerate(batches, start=1):
                step += 1
                cur_lr = self.schedule.rate(step, total_steps)

                t0 = time.perf_counter()
                training_batch = [self.case_transform.transform(c) for c in batch]
                scores = self.model.score(training_batch, train=True)
                targets = torch.as_tensor([c.target_index() for c in training_batch], device=scores.device)
                loss_tensor = self.loss.compute(scores, targets)

                self.learner.learn(loss_tensor, lr=cur_lr)
                dt = (time.perf_counter() - t0) * 1000.0
                total_loss += loss_tensor.item()
                 
                val_loss = self.evaluate(val_cases) if batch_index in (total_batches // 2, total_batches) else None 

                stats = TrainingStats(
                    epoch=ep,
                    total_epochs=epochs,
                    step=step,
                    total_steps=total_steps,
                    train_loss=total_loss / batch_index,
                    val_loss=val_loss,
                    lr=cur_lr,
                    elapsed_ms=dt,
                    batch=batch_index,
                    total_batches=total_batches,
                )

                logger.info(
                    "Epoch %d/%d | Batch %d/%d | Train Loss: %.4f%s | LR: %.2e | %.1fms",
                    ep, epochs, batch_index, total_batches, stats.train_loss,
                    f" | Val Loss: {val_loss:.4f}" if val_loss is not None else "",
                    cur_lr, dt,
                )

                yield stats

            self.learner.update()

    def __or__(self, layer: ITrainerLayer) -> "ITrainer":
        return layer(self)
