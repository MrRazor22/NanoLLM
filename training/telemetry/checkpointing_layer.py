from pathlib import Path
from typing import Callable, Iterator, Optional, Union

from training.epoch_trainer import ITrainer
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

    def __call__(self, inner: ITrainer) -> "CheckpointingLayer":
        self.inner = inner
        return self

    attach = __call__

    def fit(self, train_data, epochs: int = 1, **kwargs) -> Iterator[StepStats]:
        if not self.inner:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        for s in self.inner.fit(train_data, epochs=epochs, **kwargs):
            if self.output_path and s.step == s.total_steps and hasattr(self.inner, "model"):
                from model.weights import WeightMeta
                target = self.output_path
                meta = WeightMeta(epoch=s.epoch, train_loss=s.loss, val_loss=s.val_loss)
                if str(target).endswith(".pt") or "/" in str(target) or "\\" in str(target):
                    self.inner.model.save(target, meta=meta)
                else:
                    self.inner.model.save(f"{target}_v{s.epoch}.pt", meta=meta)
                    self.inner.model.save(f"{target}.pt", meta=meta)
            yield s

    def evaluate(self, cases, on_progress=None) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        return self.inner.evaluate(cases, on_progress=on_progress)

    def __or__(self, layer: Callable[[ITrainer], ITrainer]) -> ITrainer:
        return layer(self)
