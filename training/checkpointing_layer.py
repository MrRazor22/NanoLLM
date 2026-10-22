from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

from data.case import DecisionCase
from training.epoch_trainer import ITrainer, ITrainerLayer, TrainingStats

class CheckpointingLayer(ITrainer):
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

    def train(
        self,
        train_cases: Sequence[DecisionCase],
        val_cases: Sequence[DecisionCase],
        epochs: int = 1,
    ) -> Iterator[TrainingStats]:
        if not self.inner:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        model = self.model
        if self.output_path and model is None:
            raise RuntimeError("CheckpointingLayer cannot locate model in trainer pipeline.")
        for s in self.inner.train(train_cases, val_cases, epochs=epochs):
            if self.output_path and model:
                target = self.output_path
                p_target = Path(target)
                if s.val_loss is not None and s.val_loss < self.best_val_loss:
                    self.best_val_loss = s.val_loss
                    model.save(target, info=s)
                if s.batch == s.total_batches:
                    epoch_target = p_target if (p_target.suffix or p_target.parent != Path(".")) else Path(f"{target}_v{s.epoch}")
                    model.save(epoch_target, info=s)
            yield s

    def evaluate(self, cases: Sequence[DecisionCase]) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        return self.inner.evaluate(cases)

    def __or__(self, layer: ITrainerLayer) -> ITrainer:
        return layer(self)
