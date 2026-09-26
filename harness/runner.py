from pathlib import Path
from typing import Any, Optional, Union
import torch

from pipeline import PipelineComposable
from nanollm.training import CheckpointingLayer, EpochTrainer, ITrainer, MetricsLayer
from harness.dataset import TrainingDataset

class TrainingRunner(PipelineComposable):
    """Training Primitive: Coordinates hardware setup, dataset loaders, and trainer execution."""

    def __init__(self, trainer: ITrainer, backbone: str = "answerdotai/ModernBERT-base"):
        self.trainer = trainer
        self.backbone = backbone

    @classmethod
    def from_backbone(
        cls,
        backbone_name: str = "answerdotai/ModernBERT-base",
        lr: float = 1.5e-5,
        accum_steps: int = 2,
        init_checkpoint: Optional[Union[str, Path]] = None,
        checkpoint_output: Optional[Union[str, Path]] = None,
        metrics_output: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ) -> "TrainingRunner":
        trainer: ITrainer = EpochTrainer.from_backbone(
            backbone_name=backbone_name,
            lr=lr,
            accum_steps=accum_steps,
            init_checkpoint=str(init_checkpoint) if init_checkpoint else None,
            device=device,
        )
        if metrics_output:
            trainer = trainer | MetricsLayer(output_path=metrics_output, sink=print)
        if checkpoint_output:
            trainer = trainer | CheckpointingLayer(output_path=checkpoint_output)

        return cls(trainer, backbone=backbone_name)

    def fit(
        self,
        train_data: Union[str, Path],
        val_data: Optional[Union[str, Path]] = None,
        epochs: int = 1,
        batch_size: int = 16,
        max_tokens: int = 4000,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
    ) -> float:
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        train_loader, val_loader = TrainingDataset.loaders(
            train_data=str(train_data),
            val_data=str(val_data) if val_data else None,
            backbone=self.backbone,
            batch_size=batch_size,
            max_tokens=max_tokens,
            cache_dir=str(cache_dir) if cache_dir else None,
            seed=seed,
        )
        return self.trainer.fit(train_loader, val_loader=val_loader, epochs=epochs)

__all__ = ["TrainingRunner"]
