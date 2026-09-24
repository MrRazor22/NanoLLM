from pathlib import Path
from typing import Any, Dict
import argparse
import json
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from nanollm import (
    CalibratedLoss,
    ModelConfig,
    MultiQuestionCollator,
    NanoModel,
    SubwordTokenizer,
    load_jsonl,
)

def evaluate(model: NanoModel, dataloader: DataLoader, loss_fn: CalibratedLoss, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["mask"].to(device)
                scores = model(input_ids, mask)
                loss = loss_fn(scores, batch["meta"], device)
                total_loss += loss.item()
    return total_loss / max(1, len(dataloader))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] Initializing on: {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    backbone_name = "answerdotai/ModernBERT-base"
    tokenizer = SubwordTokenizer(backbone_name)
    collator = MultiQuestionCollator(tokenizer)

    parser = argparse.ArgumentParser(description="Train NanoLLM Foundation Decision Engine")
    parser.add_argument("--output", default=str(ROOT_DIR / "checkpoints" / "checkpoint.pt"), help="Output path for best checkpoint")
    parser.add_argument("--init", default=None, help="Path to initial checkpoint to warm-start from")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train-data", default=str(ROOT_DIR / "data" / "train.jsonl"), help="Path to train data")
    parser.add_argument("--val-data", default=str(ROOT_DIR / "data" / "val.jsonl"), help="Path to val data")
    parser.add_argument("--max-samples", type=int, default=0, help="Cap train samples for fast smoke test (0 = all)")
    args, _ = parser.parse_known_args()

    train_samples = load_jsonl(args.train_data)
    val_samples = load_jsonl(args.val_data)
    if args.max_samples > 0:
        train_samples = train_samples[:args.max_samples]
        val_samples = val_samples[:min(len(val_samples), max(100, args.max_samples // 5))]

    micro_batch, accum_steps = 8, 4
    train_loader = DataLoader(train_samples, batch_size=micro_batch, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_samples, batch_size=micro_batch, shuffle=False, collate_fn=collator)

    backbone = AutoModel.from_pretrained(backbone_name)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, hidden_dim=768, num_layers=22, num_heads=12)
    model = NanoModel(config, backbone=backbone).to(device)
    if args.init and Path(args.init).exists():
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"[INIT] Warm-started weights from {args.init}", flush=True)
    loss_fn = CalibratedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_out = args.output
    epochs = args.epochs
    best_val_loss = float("inf")
    total_steps = len(train_loader)
    print(f"[DATA] Train: {len(train_samples)} samples | Val: {len(val_samples)} samples | MicroBatch: {micro_batch} | Accum: {accum_steps} | Steps/Epoch: {total_steps}\n", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.perf_counter()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                scores = model(input_ids, mask)
                loss = loss_fn(scores, batch["meta"], device) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == total_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps
            if (step + 1) % 100 == 0 or (step + 1) == total_steps:
                elapsed = time.perf_counter() - start_time
                avg_step_ms = (elapsed / (step + 1)) * 1000.0
                curr_loss = train_loss / (step + 1)
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step+1:4d}/{total_steps}] "
                    f"Loss: {curr_loss:.4f} | Speed: {avg_step_ms:.1f}ms/step",
                    flush=True
                )


        val_loss = evaluate(model, val_loader, loss_fn, device)
        epoch_sec = time.perf_counter() - start_time
        print(f"\n---> Epoch {epoch} Complete | Train Loss: {train_loss / total_steps:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_sec:.1f}s", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_out)
            meta_path = Path(checkpoint_out).with_suffix(".meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"checkpoint": str(checkpoint_out), "epoch": epoch, "val_loss": val_loss, "train_loss": train_loss / total_steps, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)
            print(f"     [CHECKPOINT] Saved best model to {checkpoint_out} (val_loss: {val_loss:.4f})\n", flush=True)

    print(f"[DONE] Training finished. Best Val Loss: {best_val_loss:.4f}", flush=True)

if __name__ == "__main__":
    main()
