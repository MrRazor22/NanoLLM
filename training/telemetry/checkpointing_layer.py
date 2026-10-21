from pathlib import Path
import time
from typing import Iterator, Optional, Sequence, Union

from data.case import DecisionCase
from training.epoch_trainer import ITrainer, ITrainerLayer
from training.stats import StepStats

class CheckpointingLayer(ITrainer):
    """Layer Decorator (lambda: P -> P): Sovereign saving telemetry at epoch-end."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        output_path: Optional[Union[str, Path]] = None,
    ):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.best_val_loss: float = float("inf")

    @property
    def model(self):
        return getattr(self.inner, "model", None)

    def __call__(self, inner: ITrainer) -> "CheckpointingLayer":
        self.inner = inner
        return self

    def fit(
        self,
        train_cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]:
        if not self.inner:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        model = self.model
        if self.output_path and model is None:
            raise RuntimeError("CheckpointingLayer cannot locate model in trainer pipeline.")
        for s in self.inner.fit(train_cases, epochs=epochs, val_cases=val_cases):
            if self.output_path and model:
                target = self.output_path
                meta = {
                    "epoch": str(s.epoch),
                    "train_loss": f"{s.loss:.4f}",
                    "val_loss": f"{s.val_loss:.4f}" if s.val_loss is not None else "",
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                p_target = Path(target)
                if s.val_loss is not None and s.val_loss < self.best_val_loss:
                    self.best_val_loss = s.val_loss
                    model.save(target, meta=meta)
                if s.batch == s.total_batches:
                    epoch_target = p_target if (p_target.suffix or p_target.parent != Path(".")) else Path(f"{target}_v{s.epoch}")
                    model.save(epoch_target, meta=meta)
                    if s.val_loss is None:
                        model.save(target, meta=meta)
            yield s

    def evaluate(self, cases: Sequence[DecisionCase]) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        return self.inner.evaluate(cases)

    def __or__(self, layer: ITrainerLayer) -> ITrainer:
        return layer(self)
