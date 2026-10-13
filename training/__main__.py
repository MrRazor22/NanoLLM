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
    epochs: int = typer.Option(1, help="Number of training epochs"),
    batch_size: int = typer.Option(16, help="Batch size"),
    lr: float = typer.Option(1.5e-5, help="Learning rate"),
    accum_steps: int = typer.Option(2, help="Gradient accumulation steps"),
    seed: int = typer.Option(42, help="Random seed"),
    compile: bool = typer.Option(False, "--compile", help="Compile model with torch.compile"),
) -> None:
    latest_ckpt, _ = get_latest_checkpoint()
    init_ckpt = None if from_scratch or (init_checkpoint and init_checkpoint.lower() in ("none", "")) else (init_checkpoint or (str(latest_ckpt) if latest_ckpt else None))
    out_path = output or str(get_next_checkpoint_path())

    model = TorchModel(
        backbone_name=backbone,
        checkpoint_path=init_ckpt,
        lr=lr,
        accum_steps=accum_steps,
        compile=compile,
    )
    trainer: ITrainer = EpochTrainer(model=model, batch_size=batch_size)
    if metrics_output:
        trainer = trainer | MetricsLayer(output_path=metrics_output, sink=lambda msg: print(msg, flush=True), log_interval_sec=5.0)
    if out_path:
        trainer = trainer | CheckpointingLayer(output_path=out_path)

    dataset = Dataset(corpus_dir=corpus_dir)
    train_cases = dataset.load("train")
    val_cases = dataset.load("val")

    print(f"Loaded dataset: {len(train_cases)} train cases, {len(val_cases)} val cases.")
    print(f"Starting training on {backbone} for {epochs} epoch(s)...")

    stats = list(trainer.fit(train_cases, epochs=epochs, val_cases=val_cases))
    if stats and stats[-1].val_loss is not None:
        print(f"Final Validation Loss: {stats[-1].val_loss:.4f}")

    print(f"Training completed. Weights saved to: {out_path}")

if __name__ == "__main__":
    app()
