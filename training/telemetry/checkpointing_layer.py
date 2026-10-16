from pathlib import Path
from typing import Iterator, Optional, Union

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

    def fit(self, train_data, epochs: int = 1, **kwargs) -> Iterator[StepStats]:
        if not self.inner:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        model = self.model
        if self.output_path and model is None:
            raise RuntimeError("CheckpointingLayer cannot locate model in trainer pipeline.")
        for s in self.inner.fit(train_data, epochs=epochs, **kwargs):
            if self.output_path and s.batch == s.total_batches and model:
                from model.weights import WeightMeta
                target = self.output_path
                meta = WeightMeta(epoch=s.epoch, train_loss=s.loss, val_loss=s.val_loss)
                p_target = Path(target)
                if p_target.suffix or p_target.parent != Path("."):
                    model.save(p_target, meta=meta)
                else:
                    model.save(f"{target}_v{s.epoch}", meta=meta)
                    if s.val_loss is None or s.val_loss < self.best_val_loss:
                        if s.val_loss is not None:
                            self.best_val_loss = s.val_loss
                        model.save(target, meta=meta)
            yield s

    def evaluate(self, cases) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        return self.inner.evaluate(cases)

    def __or__(self, layer: ITrainerLayer) -> ITrainer:
        return layer(self)
