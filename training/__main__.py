import logging
import random
import time
from typing import Optional
import torch
import typer

from data.dataset import Dataset
from model.device import seed_device
from model.torch_model import TorchModel
from training.checkpointing_layer import CheckpointingLayer
from training.epoch_trainer import EpochTrainer, ITrainer
from training.policies.learner import AdamWLearner
from training.policies.schedule import CosineSchedule
from training.policies.transform import ShuffleCandidates

app = typer.Typer(add_completion=False, help="Piqo Model Trainer CLI")
logger = logging.getLogger("piqo.training")

@app.command()
def main(
    backbone: str = typer.Option("answerdotai/ModernBERT-base", help="HuggingFace backbone model"),
    name: str = typer.Option("piqo", "--name", "-n", help="Target weights name/prefix"),
    init_weights: Optional[str] = typer.Option(None, "--init", help="Initial weights to resume from"),
    epochs: int = typer.Option(1, help="Number of training epochs"),
    batch_size: int = typer.Option(16, help="Batch size"),
    lr: float = typer.Option(1e-5, help="Base learning rate"),
    accum_steps: int = typer.Option(2, help="Gradient accumulation steps"),
    seed: int = typer.Option(42, help="Random seed"),
    compile: bool = typer.Option(False, "--compile", help="Compile model with torch.compile"),
    warmup_ratio: float = typer.Option(0.05, help="Warmup ratio for cosine schedule"),
    eval_test: bool = typer.Option(False, "--eval-test", help="Evaluate internal holdout test split after training"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    random.seed(seed)
    seed_device(seed)

    model = TorchModel(backbone=backbone)
    if init_weights:
        model.load(init_weights)
    if compile:
        model = torch.compile(model)

    splits = Dataset().load()
    schedule_policy = CosineSchedule(base_lr=lr, warmup_ratio=warmup_ratio)
    learner_policy = AdamWLearner(model.parameters(), lr=lr, accum_steps=accum_steps)
    transform_policy = ShuffleCandidates(seed=seed)
    trainer: ITrainer = (
        EpochTrainer(
            model=model,
            batch_size=batch_size,
            learner=learner_policy,
            schedule=schedule_policy,
            case_transform=transform_policy,
        )
        | CheckpointingLayer(output_path=name)
    )

    logger.info(
        "Starting training: %s | %d epochs | lr: %.2e | %d train, %d val cases",
        backbone, epochs, lr, len(splits.train), len(splits.val),
    )
    t0 = time.perf_counter()
    stats = list(trainer.train(splits.train, splits.val, epochs=epochs))
    total_time = time.perf_counter() - t0

    last_val = stats[-1].val_loss if stats and stats[-1].val_loss is not None else 0.0
    logger.info("Training completed in %.1fs | Final Val Loss: %.4f", total_time, last_val)

    if eval_test:
        test_loss = trainer.evaluate(splits.test)
        logger.info("Test holdout evaluation: %.4f loss across %d cases", test_loss, len(splits.test))
