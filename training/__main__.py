import random
import time
from typing import Optional
from rich import box
from rich.console import Console
from rich.table import Table
import torch
import transformers
import typer

transformers.logging.set_verbosity_error()

from data.dataset import Dataset
from model.torch_model import TorchModel
from model.weights import list_weights
from training.epoch_trainer import EpochTrainer, ITrainer
from training.policies.schedule import ConstantSchedule, CosineSchedule
from training.telemetry.checkpointing_layer import CheckpointingLayer
from training.telemetry.metrics_layer import MetricsLayer

DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"

app = typer.Typer(add_completion=False, help="Piqo Model Trainer CLI")

def make_card(title: str, rows: list[tuple[str, str]], style: str = "bold magenta") -> Table:
    table = Table(title=title, box=box.HORIZONTALS, title_style=style)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold green" if style == "bold white" else "green")
    for k, v in rows:
        table.add_row(k, v)
    return table

@app.command()
def main(
    backbone: str = typer.Option(DEFAULT_BACKBONE, help="HuggingFace backbone model"),
    name: str = typer.Option("piqo", "--name", "-n", help="Target weights name/prefix (default: 'piqo')"),
    init_weights: Optional[str] = typer.Option(None, "--init", help="Initial weights to resume from"),
    from_scratch: bool = typer.Option(False, "--from-scratch", help="Train from raw base backbone"),
    epochs: Optional[int] = typer.Option(None, help="Number of training epochs (default: 2 for --from-scratch, 1 for adaptation)"),
    batch_size: int = typer.Option(16, help="Batch size"),
    lr: Optional[float] = typer.Option(None, help="Learning rate (default: 3e-5 for --from-scratch, 1e-5 for adaptation)"),
    accum_steps: int = typer.Option(2, help="Gradient accumulation steps"),
    seed: int = typer.Option(42, help="Random seed"),
    compile: bool = typer.Option(False, "--compile", help="Compile model with torch.compile"),
    schedule: str = typer.Option("cosine", help="Learning rate schedule policy ('cosine', 'constant')"),
    warmup_ratio: float = typer.Option(0.05, help="Warmup ratio for cosine schedule"),
    eval_test: bool = typer.Option(False, "--eval-test", help="Evaluate internal holdout test split after training"),
) -> None:
    console = Console()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    has_piqo = any(w.name == "piqo" for w in list_weights())
    init_ckpt = None if from_scratch or (init_weights and init_weights.lower() in ("none", "")) else (init_weights or ("piqo" if has_piqo else None))
    out_target = name

    target_epochs = epochs if epochs is not None else (2 if from_scratch else 1)
    target_lr = lr if lr is not None else (3e-5 if from_scratch else 1e-5)

    model = TorchModel(backbone_name=backbone, weights=init_ckpt, accum_steps=accum_steps, compile=compile)

    schedule_policy = ConstantSchedule() if schedule.lower() == "constant" else CosineSchedule(warmup_ratio=warmup_ratio)
    trainer: ITrainer = (
        EpochTrainer(model=model, batch_size=batch_size, base_lr=target_lr, schedule=schedule_policy)
        | MetricsLayer(log_interval_sec=5.0)
        | CheckpointingLayer(output_path=out_target)
    )

    dataset = Dataset()
    splits = dataset.load()
    train_cases = splits.train
    val_cases = splits.val

    init_desc = f"Checkpoint ({init_ckpt})" if init_ckpt else "Scratch (Raw Backbone + Xavier Head)"
    console.print(f"\n[bold cyan]Piqo Model Trainer[/bold cyan] | Target: [bold green]model/weights/{out_target}[/bold green]\n")
    console.print(make_card("Training Configuration", [
        ("Backbone", backbone),
        ("Initialization", init_desc),
        ("Epochs", str(target_epochs)),
        ("Batch Size (Effective)", f"{batch_size} (Effective: {batch_size * accum_steps})"),
        ("Learning Rate", f"{target_lr:.2e} ({schedule})"),
        ("Dataset Splits", f"{len(train_cases):,} train, {len(val_cases):,} val cases"),
    ]))
    console.print("")

    stats = list(trainer.fit(train_cases, epochs=target_epochs, val_cases=val_cases))

    res = [("Status", "Completed Successfully")]
    if stats:
        res.append(("Final Train Loss", f"{stats[-1].loss:.4f}"))
        if stats[-1].val_loss is not None:
            res.append(("Final Val Loss", f"{stats[-1].val_loss:.4f}"))

    if eval_test:
        test_cases = splits.test
        console.print(f"\n[dim]Evaluating internal holdout test split ({len(test_cases):,} cases)...[/dim]")
        t_eval = time.perf_counter()
        test_loss = trainer.evaluate(test_cases)
        ms = (time.perf_counter() - t_eval) * 1000.0 / max(1, len(test_cases))
        res.append(("Test Holdout Loss", f"{test_loss:.4f} ({ms:.2f}ms/case)"))

    res.append(("Output Weights", f"model/weights/{out_target}"))
    console.print(make_card("Training Results", res, style="bold white"))
    console.print("")

if __name__ == "__main__":
    app()
