import argparse
from pathlib import Path
import torch

from nanollm.inference.assembler_policy import SlotAssembler
from nanollm.model import NanoModel
from nanollm.training import CalibratedLoss, EpochTrainer
from training.dataset import TrainingDataset
from training.layers import CheckpointingLayer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "train_adapt.jsonl"
DEFAULT_VAL = DEFAULT_DATA_DIR / "val_adapt.jsonl"
DEFAULT_OUTPUT = ROOT / "checkpoints" / "checkpoint_trained.pt"
DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"

def main() -> None:
    parser = argparse.ArgumentParser(description="Train NanoLLM")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, help="HuggingFace backbone model")
    parser.add_argument("--train-data", type=str, default=str(DEFAULT_TRAIN), help="Path to training jsonl")
    parser.add_argument("--val-data", type=str, default=str(DEFAULT_VAL), help="Path to validation jsonl")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output checkpoint path")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Initial checkpoint path")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4000, help="Max tokens per batch for dynamic batching")
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log-interval", type=float, default=10.0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assembler = SlotAssembler(args.backbone)
    pin = device.type == "cuda"

    # 1. Dataset primitives loaded with assembler
    train_dataset = TrainingDataset.from_jsonl(args.train_data, assembler=assembler)
    val_dataset = TrainingDataset.from_jsonl(args.val_data, assembler=assembler)

    # 2. Dataset yields PyTorch DataLoaders directly with seed passed from CLI
    train_loader = train_dataset.get_loader(
        batch_size=args.batch_size, max_tokens=args.max_tokens, shuffle=True, pin_memory=pin, seed=args.seed
    )
    val_loader = val_dataset.get_loader(
        batch_size=args.batch_size, max_tokens=args.max_tokens, shuffle=False, pin_memory=pin, seed=args.seed
    )

    # 3. Model & Optimizer
    model = NanoModel.from_backbone(args.backbone, vocab_size=assembler.tokenizer.vocab_size).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=pin)
    loss_fn = CalibratedLoss()

    # 4. Trainer Primitive wrapped with Checkpointing Layer
    trainer = EpochTrainer(
        model, optimizer, loss_fn, device, accum_steps=args.accum_steps, log_interval=args.log_interval
    ) | CheckpointingLayer(output_path=args.output)

    # 5. Fit model directly on DataLoaders
    print(f"Starting training on {device} ({len(train_dataset)} train samples, {len(val_dataset)} val samples)...")
    trainer.fit(train_loader, val_loader, epochs=args.epochs)

if __name__ == "__main__":
    main()
