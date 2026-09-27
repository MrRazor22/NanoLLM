from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
import torch
from pipeline import PipelineComposable
from nanollm.training.epoch_trainer import Batch, ITrainer, StepStats

class CheckpointingLayer(PipelineComposable, ITrainer):
    """ATA Layer: saves checkpoint at each epoch-end (when val_loss is populated in the stream)."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        output_path: Optional[str] = None,
        save_optimizer: bool = False,
    ):
        if inner is not None and not isinstance(inner, ITrainer):
            raise TypeError(f"inner must implement ITrainer, got {type(inner).__name__}")
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.save_optimizer = save_optimizer
        self.best_val_loss = float("inf")

    def attach(self, inner: ITrainer) -> "CheckpointingLayer":
        if not isinstance(inner, ITrainer):
            raise TypeError(f"inner must implement ITrainer, got {type(inner).__name__}")
        self.inner = inner
        return self

    @property
    def model(self) -> Any:
        return getattr(self.inner, "model")

    def _save(self, path: Path, stats: Optional[StepStats] = None) -> None:
        import json, time
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": path.name,
            "epoch": stats.epoch if stats else 1,
            "train_loss": round(stats.loss, 6) if stats else None,
            "val_loss": round(stats.val_loss, 6) if stats and stats.val_loss is not None else None,
            "lr": stats.lr if stats else None,
            "step": stats.step if stats else None,
            "total_steps": stats.total_steps if stats else None,
        }
        payload = {"model_state_dict": self.model.state_dict(), "meta": meta}
        if self.save_optimizer and hasattr(self.inner, "optimizer"):
            payload["optimizer_state_dict"] = getattr(self.inner, "optimizer").state_dict()
        torch.save(payload, str(path))

        # Update persistent manifest
        manifest_path = path.parent / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        manifest[path.name] = meta
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def evaluate(self, batches: Iterable[Batch]) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        return self.inner.evaluate(batches)

    def fit(
        self,
        train_batches: Iterable[Batch],
        val_batches: Optional[Iterable[Batch]] = None,
        epochs: int = 1,
    ) -> Iterator[StepStats]:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        for s in self.inner.fit(train_batches, val_batches=val_batches, epochs=epochs):
            if s.val_loss is not None and self.output_path:
                if s.val_loss < self.best_val_loss:
                    self.best_val_loss = s.val_loss
                    self._save(self.output_path, stats=s)
            yield s

__all__ = ["CheckpointingLayer"]
