from pathlib import Path
from typing import Optional
import transformers
import typer

transformers.logging.set_verbosity_error()

from data.dataset import CORPUS_DIR, Dataset
from model.layouts.checkpoints import get_latest_checkpoint, get_next_checkpoint_path
from model.torch_model import TorchModel
from training.epoch_trainer import EpochTrainer, ITrainer
from training.telemetry.checkpointing_layer import CheckpointingLayer
from training.telemetry.metrics_layer import MetricsLayer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "weights" / "metrics.json"
DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"

app = typer.Typer(add_completion=False, help="Piqo Model Trainer CLI")

@app.command()
def main(
    backbone: str = typer.Option(DEFAULT_BACKBONE, help="HuggingFace backbone model"),
    corpus_dir: str = typer.Option(str(CORPUS_DIR), help="Path to corpus directory"),
    output: Optional[str] = typer.Option(None, help="Output weight checkpoint path"),
    metrics_output: str = typer.Option(str(DEFAULT_METRICS), help="Metrics JSON output path"),
    init_checkpoint: Optional[str] = typer.Option(None, help="Initial weights path"),
    from_scratch: bool = typer.Option(False, "--from-scratch", help="Train from raw base backbone"),
    epochs: Optional[int] = typer.Option(None, help="Number of training epochs (default: 2 for --from-scratch, 1 for adaptation)"),
    batch_size: int = typer.Option(16, help="Batch size"),
    lr: Optional[float] = typer.Option(None, help="Learning rate (default: 3e-5 for --from-scratch, 1e-5 for adaptation)"),
    accum_steps: int = typer.Option(1, help="Gradient accumulation steps"),
    seed: int = typer.Option(42, help="Random seed"),
    compile: bool = typer.Option(False, "--compile", help="Compile model with torch.compile"),
    schedule: str = typer.Option("cosine", help="Learning rate schedule policy ('cosine', 'constant')"),
    warmup_ratio: float = typer.Option(0.05, help="Warmup ratio for cosine schedule"),
) -> None:
    latest_ckpt, _ = get_latest_checkpoint()
    init_ckpt = None if from_scratch or (init_checkpoint and init_checkpoint.lower() in ("none", "")) else (init_checkpoint or (str(latest_ckpt) if latest_ckpt else None))
    out_path = output or str(get_next_checkpoint_path())

    target_epochs = epochs if epochs is not None else (2 if from_scratch else 1)
    target_lr = lr if lr is not None else (3e-5 if from_scratch else 1e-5)

    model = TorchModel(
        backbone_name=backbone,
        checkpoint_path=init_ckpt,
        lr=target_lr,
        accum_steps=accum_steps,
        compile=compile,
    )
    from training.policies.schedule_policy import ConstantSchedulePolicy, CosineSchedulePolicy
    sched_policy = ConstantSchedulePolicy() if schedule.lower() == "constant" else CosineSchedulePolicy(warmup_ratio=warmup_ratio)
    trainer: ITrainer = EpochTrainer(model=model, batch_size=batch_size, schedule_policy=sched_policy)
    if metrics_output:
        trainer = trainer | MetricsLayer(output_path=metrics_output, sink=lambda msg: print(msg, flush=True), log_interval_sec=5.0)
    if out_path:
        trainer = trainer | CheckpointingLayer(output_path=out_path)

    dataset = Dataset(corpus_dir=corpus_dir)
    train_cases = dataset.load("train")
    val_cases = dataset.load("val")

    from collections import Counter
    train_cats = Counter(c.category or "general" for c in train_cases)
    cat_summary = ", ".join(f"{cat}: {cnt}" for cat, cnt in train_cats.most_common())
    print(f"Loaded dataset: {len(train_cases)} train, {len(val_cases)} val cases.")
    print(f"Domain Distribution: {cat_summary}")
    if init_ckpt:
        print(f"Initialized from checkpoint: {init_ckpt}")
    else:
        print("Initialized from scratch: raw ModernBERT backbone with fresh Xavier head (zero checkpoint weights).")
    print(f"Starting training on {backbone} (LR: {target_lr}, Batch: {batch_size}) for {target_epochs} epoch(s)...")

    stats = list(trainer.fit(train_cases, epochs=target_epochs, val_cases=val_cases))
    if stats and stats[-1].val_loss is not None:
        print(f"Final Validation Loss: {stats[-1].val_loss:.4f}")

    print(f"Training completed. Weights saved to: {out_path}")

if __name__ == "__main__":
    app()
