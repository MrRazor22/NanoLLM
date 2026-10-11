import random
import time
from typing import Callable, Iterator, Optional, Protocol, Sequence, Union, runtime_checkable

from inference.case import DecisionCase
from model import IModel, TorchModel
from training.batches import StepStats
from training.checkpoints import CheckpointMeta, save_checkpoint
from training.policies.batch_collator import IBatchCollator, PaddedBatchCollator

@runtime_checkable
class ITrainer(Protocol):
    """Core Primitive: Operational boundary for model optimization and checkpoint persistence."""
    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]: ...
    def save(self, meta: Optional[CheckpointMeta] = None) -> None: ...

class EpochTrainer(ITrainer):
    """Core Primitive Implementation: Supervised training loop over domain decision cases."""

    def __init__(
        self,
        model: Optional[IModel] = None,
        collator: Optional[IBatchCollator] = None,
        batch_size: int = 16,
    ):
        self.model = model or TorchModel()
        self.collator = collator or PaddedBatchCollator()
        self.batch_size = batch_size


    def save(self, meta: Optional[CheckpointMeta] = None) -> None:
        save_checkpoint(self.model, meta=meta)

    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]:
        items = list(cases)
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size if items else 0

        for ep in range(1, epochs + 1):
            random.shuffle(items)
            total_loss, last_t = 0.0, time.perf_counter()

            for n, i in enumerate(range(0, len(items), self.batch_size)):
                batch = self.collator.collate(items[i : i + self.batch_size])
                step_loss = self.model.train_step(batch)
                total_loss += step_loss
                now = time.perf_counter()
                dt = (now - last_t) * 1000.0
                last_t = now

                v_loss = None
                if (n + 1 == total_batches) and val_cases:
                    v_items = list(val_cases)
                    v_total, v_count = 0.0, 0
                    for vi in range(0, len(v_items), self.batch_size):
                        v_batch = self.collator.collate(v_items[vi : vi + self.batch_size])
                        v_total += self.model.eval_step(v_batch)
                        v_count += 1
                    v_loss = v_total / max(1, v_count)

                yield StepStats(
                    epoch=ep,
                    total_epochs=epochs,
                    step=n + 1,
                    total_steps=total_batches,
                    loss=total_loss / (n + 1),
                    elapsed_ms=dt,
                    lr=0.0,
                    val_loss=v_loss,
                    batch=n + 1,
                    total_batches=total_batches,
                )

    def __or__(self, layer: Union[type, Callable[["ITrainer"], "ITrainer"]]) -> "ITrainer":
        return layer(self) if isinstance(layer, type) else (getattr(layer, "attach")(self) if hasattr(layer, "attach") else layer(self))

__all__ = ["ITrainer", "EpochTrainer"]
