import argparse
from pathlib import Path
import random
from typing import List, Sequence
import torch
from transformers import AutoTokenizer

from inference import SlotEncoder
from data import CORPUS_DIR, Dataset, DiskCachePolicy
from training import (
    CHECKPOINTS_DIR,
    DEFAULT_CHECKPOINT,
    CheckpointingLayer,
    EpochTrainer,
    ITrainer,
    MetricsLayer,
    get_latest_checkpoint,
    get_next_checkpoint_path,
)
from inference.questions import DecisionSample

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "checkpoints" / "metrics.json"
DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"

def batch_samples(samples: Sequence[DecisionSample], encoder: SlotEncoder, batch_size: int, shuffle: bool = True) -> List[dict]:
    items = list(samples)
    if shuffle:
        random.shuffle(items)
    batches = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        encoded = [encoder.encode_with_targets(s.state, s.questions)[0::2] for s in chunk]
        batches.append(encoder.collate_batch(encoded))
    return batches

def main() -> None:
    latest_ckpt, _ = get_latest_checkpoint()
    default_init = str(latest_ckpt) if latest_ckpt else None
    default_out = str(get_next_checkpoint_path())

    parser = argparse.ArgumentParser(description="Piqo Model Trainer")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, help="HuggingFace backbone model")
    parser.add_argument("--corpus-dir", type=str, default=str(CORPUS_DIR), help="Path to corpus directory")
    parser.add_argument("--output", type=str, default=default_out, help=f"Output checkpoint path (default: {default_out})")
    parser.add_argument("--metrics-output", type=str, default=str(DEFAULT_METRICS), help="Metrics JSON output path")
    parser.add_argument("--init-checkpoint", type=str, default=default_init, help=f"Initial checkpoint path (default: {default_init})")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
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

    encoder = SlotEncoder(AutoTokenizer.from_pretrained(args.backbone))
    dataset = Dataset(cache_policy=DiskCachePolicy(Path(args.corpus_dir)), seed=args.seed)

    train_samples = dataset.load("train")
    val_samples = dataset.load("val")

    print(f"Loaded dataset: {len(train_samples)} train samples, {len(val_samples)} val samples.")

    train_batches = batch_samples(train_samples, encoder, batch_size=args.batch_size, shuffle=True)
    val_batches = batch_samples(val_samples, encoder, batch_size=args.batch_size, shuffle=False) if val_samples else []

    print(f"Starting training: {len(train_batches)} train batches, {len(val_batches)} val batches...")
    list(trainer.fit(train_batches, epochs=args.epochs))

    if val_batches:
        val_loss = trainer.evaluate(val_batches)
        print(f"Final Validation Loss: {val_loss:.4f}")

    print(f"Training completed. Checkpoint saved to: {args.output}")

if __name__ == "__main__":
    main()
