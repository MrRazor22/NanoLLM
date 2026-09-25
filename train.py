import argparse
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from nanollm.inference.policies.tokenizer import SubwordTokenizer
from nanollm.model import ModelConfig, NanoModel
from nanollm.training import (
    CalibratedLoss,
    CheckpointingLayer,
    EpochTrainer,
    MultiQuestionCollator,
)
from nanollm.training.policies.dataset import load_jsonl

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = ROOT / "nanollm" / "training" / "data" / "train_adapt.jsonl"
DEFAULT_VAL = ROOT / "nanollm" / "training" / "data" / "val_adapt.jsonl"
DEFAULT_OUTPUT = ROOT / "nanollm" / "model" / "checkpoints" / "checkpoint_trained.pt"

def main() -> None:
    parser = argparse.ArgumentParser(description="Train NanoLLM")
    parser.add_argument("--train-data", type=str, default=str(DEFAULT_TRAIN))
    parser.add_argument("--val-data", type=str, default=str(DEFAULT_VAL))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Initial checkpoint to start adaptation from")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SubwordTokenizer("answerdotai/ModernBERT-base")
    collator = MultiQuestionCollator(tokenizer)

    train_data = load_jsonl(args.train_data)
    val_data = load_jsonl(args.val_data)
    if args.max_samples > 0:
        train_data = train_data[:args.max_samples]
        val_data = val_data[:max(50, args.max_samples // 5)]

    train_idx = sorted(range(len(train_data)), key=lambda i: len(train_data[i].state))
    train_batches = [train_idx[i:i + args.batch_size] for i in range(0, len(train_idx), args.batch_size)]
    random.Random(42).shuffle(train_batches)

    val_idx = sorted(range(len(val_data)), key=lambda i: len(val_data[i].state))
    val_batches = [val_idx[i:i + args.batch_size] for i in range(0, len(val_idx), args.batch_size)]

    pin = device.type == "cuda"
    train_loader = DataLoader(train_data, batch_sampler=train_batches, collate_fn=collator, pin_memory=pin)
    val_loader = DataLoader(val_data, batch_sampler=val_batches, collate_fn=collator, pin_memory=pin)

    backbone = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
    config = ModelConfig(vocab_size=tokenizer.vocab_size, hidden_dim=768, num_layers=22, num_heads=12)
    model = NanoModel(config, backbone=backbone).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"Loaded initial weights from {args.init_checkpoint}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = CalibratedLoss()

    trainer = EpochTrainer(
        model, optimizer, loss_fn, device, accum_steps=args.accum_steps, log_interval=args.log_interval
    ) | CheckpointingLayer(output_path=args.output, val_loader=val_loader)

    print(f"Starting training on {device} ({len(train_data)} train samples, {len(val_data)} val samples)...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {loss:.4f} | Best Val Loss: {trainer.best_val_loss:.4f} | Elapsed: {time.perf_counter() - t0:.1f}s")

if __name__ == "__main__":
    main()
