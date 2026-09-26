import argparse
from pathlib import Path
import torch

from nanollm.training import CheckpointingLayer, EpochTrainer, ITrainer, MetricsLayer
from harness.dataset import TrainingDataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "splits" / "train.jsonl"
DEFAULT_VAL = DEFAULT_DATA_DIR / "splits" / "val.jsonl"
DEFAULT_OUTPUT = ROOT / "checkpoints" / "checkpoint_trained.pt"
DEFAULT_METRICS = ROOT / "checkpoints" / "metrics.json"
DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"

def main() -> None:
    parser = argparse.ArgumentParser(description="Train NanoLLM")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, help="HuggingFace backbone model")
    parser.add_argument("--train-data", type=str, default=str(DEFAULT_TRAIN), help="Path to training jsonl")
    parser.add_argument("--val-data", type=str, default=str(DEFAULT_VAL), help="Path to validation jsonl")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output checkpoint path")
    parser.add_argument("--metrics-output", type=str, default=str(DEFAULT_METRICS), help="Metrics JSON output path")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Initial checkpoint path")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4000, help="Max tokens per batch for dynamic batching")
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    trainer: ITrainer = EpochTrainer.from_backbone(
        backbone_name=args.backbone,
        lr=args.lr,
        accum_steps=args.accum_steps,
        init_checkpoint=args.init_checkpoint,
    )
    if args.metrics_output:
        trainer = trainer | MetricsLayer(output_path=args.metrics_output, sink=print)
    if args.output:
        trainer = trainer | CheckpointingLayer(output_path=args.output)

    train_batches, val_batches = TrainingDataset.loaders(
        train_data=args.train_data,
        val_data=args.val_data,
        backbone=args.backbone,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    for _ in trainer.fit(train_batches, val_batches=val_batches, epochs=args.epochs):
        pass

if __name__ == "__main__":
    main()

