import argparse
from pathlib import Path
import torch

from nanollm import (
    CheckpointingLayer,
    EpochTrainer,
    ITrainer,
    MetricsLayer,
    get_latest_checkpoint,
    get_next_checkpoint_path,
)
from harness.dataset import TrainingDataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "splits" / "train.jsonl"
DEFAULT_VAL = DEFAULT_DATA_DIR / "splits" / "val.jsonl"
DEFAULT_OUTPUT = ROOT / "checkpoints" / "checkpoint_trained.pt"
DEFAULT_METRICS = ROOT / "checkpoints" / "metrics.json"
DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"

def main() -> None:
    latest_ckpt, latest_ver = get_latest_checkpoint()
    default_init = str(latest_ckpt) if latest_ckpt else None
    default_out = str(get_next_checkpoint_path())

    parser = argparse.ArgumentParser(description="Train NanoLLM")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, help="HuggingFace backbone model")
    parser.add_argument("--train-data", type=str, default=str(DEFAULT_TRAIN), help="Path to training jsonl")
    parser.add_argument("--val-data", type=str, default=str(DEFAULT_VAL), help="Path to validation jsonl")
    parser.add_argument("--output", type=str, default=default_out, help=f"Output checkpoint path (default: {default_out})")
    parser.add_argument("--metrics-output", type=str, default=str(DEFAULT_METRICS), help="Metrics JSON output path")
    parser.add_argument("--init-checkpoint", type=str, default=default_init, help=f"Initial checkpoint path (default: {default_init})")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=3600, help="Max tokens per batch for dynamic batching")
    parser.add_argument("--lr", type=float, default=1.2e-5)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile for peak CUDA speed")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    trainer: ITrainer = EpochTrainer.from_backbone(
        backbone_name=args.backbone,
        lr=args.lr,
        accum_steps=args.accum_steps,
        init_checkpoint=args.init_checkpoint,
        compile=args.compile,
    )
    if args.metrics_output:
        trainer = trainer | MetricsLayer(output_path=args.metrics_output, sink=lambda msg: print(msg, flush=True), log_interval_sec=5.0)
    if args.output:
        trainer = trainer | CheckpointingLayer(output_path=args.output)

    print(f"Preparing datasets (Train: {args.train_data}, Val: {args.val_data})...", flush=True)
    train_batches, val_batches = TrainingDataset.loaders(
        train_data=args.train_data,
        val_data=args.val_data,
        backbone=args.backbone,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(f"Batches prepared: {len(train_batches)} train batches, {len(val_batches) if val_batches is not None else 0} val batches.", flush=True)
    print(f"Beginning training for {args.epochs} epoch(s)...", flush=True)

    for _ in trainer.fit(train_batches, val_batches=val_batches, epochs=args.epochs):
        pass

if __name__ == "__main__":
    main()

