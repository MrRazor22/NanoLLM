import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from nanollm import (
    CalibratedLoss,
    ModelConfig,
    MultiQuestionCollator,
    NanoModel,
    SubwordTokenizer,
)
from nanollm.dataset import load_jsonl

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            positions = batch["question_positions"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)
            types = batch["types"]

            logits = model(input_ids, positions, mask)
            loss = loss_fn(logits, targets, types)
            total_loss += loss.item()
    return total_loss / max(1, len(dataloader))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] Initializing on: {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tokenizer = SubwordTokenizer("bert-base-uncased")
    collator = MultiQuestionCollator(tokenizer)

    train_samples = load_jsonl("data/train.jsonl")
    val_samples = load_jsonl("data/val.jsonl")

    train_loader = DataLoader(train_samples, batch_size=32, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_samples, batch_size=32, shuffle=False, collate_fn=collator)

    backbone = AutoModel.from_pretrained("bert-base-uncased")
    config = ModelConfig(vocab_size=tokenizer.vocab_size, hidden_dim=768, num_layers=12, num_heads=12, max_choices=16)
    model = NanoModel(config, backbone=backbone).to(device)
    loss_fn = CalibratedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    epochs = 5
    best_val_loss = float("inf")
    total_steps = len(train_loader)
    print(f"[DATA] Train: {len(train_samples)} samples | Val: {len(val_samples)} samples | Batch: 32 | Steps/Epoch: {total_steps}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.perf_counter()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            positions = batch["question_positions"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)
            types = batch["types"]

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(input_ids, positions, mask)
                loss = loss_fn(logits, targets, types)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            if (step + 1) % 100 == 0 or (step + 1) == total_steps:
                elapsed = time.perf_counter() - start_time
                speed = (step + 1) / max(0.01, elapsed)
                eta = (total_steps - (step + 1)) / max(0.01, speed)
                pct = ((step + 1) / total_steps) * 100
                print(f"  [Epoch {epoch:02d}/{epochs:02d}] Step {step+1:03d}/{total_steps:03d} ({pct:3.0f}%) | Loss: {loss.item():.4f} | {speed:.1f} batch/s | ETA: {int(eta)}s")

        avg_train = train_loss / total_steps
        avg_val = evaluate(model, val_loader, loss_fn, device)
        saved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "checkpoint.pt")
            saved = " --> [SAVED BEST CHECKPOINT]"
        print(f"\n>>> Epoch {epoch:02d} Complete | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}{saved}\n")

if __name__ == "__main__":
    main()
