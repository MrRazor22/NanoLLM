from pathlib import Path
from typing import Any, Dict
import argparse
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

    train_path = str(ROOT_DIR / "data" / "train.jsonl")
    val_path = str(ROOT_DIR / "data" / "val.jsonl")

    train_samples = load_jsonl(train_path)
    val_samples = load_jsonl(val_path)

    train_loader = DataLoader(train_samples, batch_size=32, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_samples, batch_size=32, shuffle=False, collate_fn=collator)

    backbone = AutoModel.from_pretrained(backbone_name)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, hidden_dim=768, num_layers=22, num_heads=12)
    model = NanoModel(config, backbone=backbone).to(device)
    loss_fn = CalibratedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    epochs = 3
    best_val_loss = float("inf")
    total_steps = len(train_loader)
    print(f"[DATA] Train: {len(train_samples)} samples | Val: {len(val_samples)} samples | Batch: 32 | Steps/Epoch: {total_steps}\n", flush=True)

    parser = argparse.ArgumentParser(description="Train NanoLLM Foundation Decision Engine")
    parser.add_argument("--output", default=str(ROOT_DIR / "checkpoint.pt"), help="Output path for best checkpoint")
    args, _ = parser.parse_known_args()
    checkpoint_out = args.output

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.perf_counter()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                scores = model(input_ids, mask)
                loss = loss_fn(scores, batch["meta"], device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            if (step + 1) % 50 == 0 or (step + 1) == total_steps:
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
            torch.save(model.state_dict(), checkpoint_out)
            print(f"     [CHECKPOINT] Saved best model to {checkpoint_out} (val_loss: {val_loss:.4f})\n", flush=True)

    print(f"[DONE] Training finished. Best Val Loss: {best_val_loss:.4f}", flush=True)

if __name__ == "__main__":
    main()
