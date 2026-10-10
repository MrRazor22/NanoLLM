from pathlib import Path
import time
from typing import TYPE_CHECKING, Iterator, Optional
import torch
from inference.questions import Composable
from training.batches import StepStats

if TYPE_CHECKING:
    from training.epoch_trainer import ITrainer

class CheckpointingLayer(Composable):
    """Layer Decorator (lambda: P -> P): Saves model checkpoint at epoch-end."""

    def __init__(self, inner: Optional["ITrainer"] = None, output_path: Optional[str] = None):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.best_val_loss = float("inf")

    def attach(self, inner: "ITrainer") -> "CheckpointingLayer":
        self.inner = inner
        return self

    @property
    def model(self):
        return getattr(self.inner, "model", None)

    @property
    def engine(self):
        return getattr(self.inner, "engine", None)

    def _save(self, path: Path, stats: Optional[StepStats] = None) -> None:
        if self.inner is None:
            return
        model = getattr(self.inner, "model", None)
        if model is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": path.name,
            "epoch": stats.epoch if stats else 1,
            "train_loss": round(stats.loss, 6) if stats else None,
            "val_loss": round(stats.val_loss, 6) if stats and stats.val_loss is not None else None,
        }
        torch.save({"model_state_dict": model.state_dict(), "meta": meta}, str(path))

    def evaluate(self, data, **kwargs) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        return self.inner.evaluate(data, **kwargs)

    def fit(self, train_data, epochs: int = 1, **kwargs) -> Iterator[StepStats]:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        for s in self.inner.fit(train_data, epochs=epochs, **kwargs):
            if self.output_path:
                self._save(self.output_path, stats=s)
            yield s

__all__ = ["CheckpointingLayer"]
