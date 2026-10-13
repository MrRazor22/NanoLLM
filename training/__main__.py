import argparse
from pathlib import Path
import transformers

transformers.logging.set_verbosity_error()

from data import CORPUS_DIR, Dataset
from model import get_latest_checkpoint, get_next_checkpoint_path
from training import CheckpointingLayer, EpochTrainer, ITrainer, MetricsLayer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "weights" / "metrics.json"
DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"

def main() -> None:
    latest_ckpt, _ = get_latest_checkpoint()
    default_init = str(latest_ckpt) if latest_ckpt else None
    default_out = str(get_next_checkpoint_path())

    parser = argparse.ArgumentParser(description="Piqo Model Trainer")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, help="HuggingFace backbone model")
    parser.add_argument("--corpus-dir", type=str, default=str(CORPUS_DIR), help="Path to corpus directory")
    parser.add_argument("--output", type=str, default=default_out, help=f"Output weight checkpoint path (default: {default_out})")
    parser.add_argument("--metrics-output", type=str, default=str(DEFAULT_METRICS), help="Metrics JSON output path")
    parser.add_argument("--init-checkpoint", type=str, default=default_init, help=f"Initial weights path (default: {default_init})")
    parser.add_argument("--from-scratch", action="store_true", help="Train from raw base backbone without loading previous checkpoints")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile for peak CUDA speed")
    args = parser.parse_args()

    init_ckpt = None if args.from_scratch or (args.init_checkpoint and args.init_checkpoint.lower() in ("none", "")) else args.init_checkpoint

    from model import TorchModel
    model = TorchModel(
        backbone_name=args.backbone,
        checkpoint_path=init_ckpt,
        lr=args.lr,
        accum_steps=args.accum_steps,
        compile=args.compile,
    )
    trainer: ITrainer = EpochTrainer(model=model, batch_size=args.batch_size)
    if args.metrics_output:
        trainer = trainer | MetricsLayer(output_path=args.metrics_output, sink=lambda msg: print(msg, flush=True), log_interval_sec=5.0)
    if args.output:
        trainer = trainer | CheckpointingLayer(output_path=args.output)

    dataset = Dataset(corpus_dir=args.corpus_dir)
    train_cases = dataset.load("train")
    val_cases = dataset.load("val")

    print(f"Loaded dataset: {len(train_cases)} train cases, {len(val_cases)} val cases.")
    print(f"Starting training on {args.backbone} for {args.epochs} epoch(s)...")

    stats = list(trainer.fit(train_cases, epochs=args.epochs, val_cases=val_cases))
    if stats and stats[-1].val_loss is not None:
        print(f"Final Validation Loss: {stats[-1].val_loss:.4f}")

    print(f"Training completed. Weights saved to: {args.output}")

if __name__ == "__main__":
    main()
