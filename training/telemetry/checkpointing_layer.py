from pathlib import Path
import time
from typing import Callable, Iterator, Optional
from model.layouts.checkpoints import CheckpointMeta
from training.stats import StepStats
from training.epoch_trainer import ITrainer

class CheckpointingLayer(ITrainer):
    """Layer Decorator (lambda: P -> P): Saves model checkpoint at epoch-end."""

    def __init__(self, inner: Optional[ITrainer] = None, output_path: Optional[str] = None):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None

    def __call__(self, inner: ITrainer) -> "CheckpointingLayer":
        self.inner = inner
        return self

    attach = __call__

    def save(self, meta: Optional[CheckpointMeta] = None) -> None:
        if self.inner is not None:
            self.inner.save(meta=meta)

    def _save(self, path: Path, stats: Optional[StepStats] = None) -> None:
        if self.inner is None:
            return
        meta = CheckpointMeta(
            path=path,
            epoch=stats.epoch if stats else 1,
            train_loss=round(stats.loss, 6) if stats else None,
            val_loss=round(stats.val_loss, 6) if stats and stats.val_loss is not None else None,
            saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.inner.save(meta=meta)

    def fit(self, train_data, epochs: int = 1, **kwargs) -> Iterator[StepStats]:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        for s in self.inner.fit(train_data, epochs=epochs, **kwargs):
            if self.output_path and s.step == s.total_steps:
                self._save(self.output_path, stats=s)
            yield s

    def __or__(self, layer: Callable[[ITrainer], ITrainer]) -> ITrainer:
        return layer(self)
