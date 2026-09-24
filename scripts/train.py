import argparse, json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from nanollm import (
    CalibratedLoss,
    EpochTrainer,
    ModelConfig,
    MultiQuestionCollator,
    NanoModel,
    SubwordTokenizer,
    load_jsonl,
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] Initializing on: {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    parser = argparse.ArgumentParser(description="Train NanoLLM Foundation Decision Engine")
    parser.add_argument("--output", default=str(ROOT_DIR / "checkpoints" / "checkpoint.pt"), help="Output path for best checkpoint")
    parser.add_argument("--init", default=None, help="Path to initial checkpoint to warm-start from")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train-data", default=str(ROOT_DIR / "data" / "train.jsonl"), help="Path to train data")
    parser.add_argument("--val-data", default=str(ROOT_DIR / "data" / "val.jsonl"), help="Path to val data")
    parser.add_argument("--max-samples", type=int, default=0, help="Cap train samples for fast smoke test (0 = all)")
    args, _ = parser.parse_known_args()

    backbone_name = "answerdotai/ModernBERT-base"
    tokenizer = SubwordTokenizer(backbone_name)
    collator = MultiQuestionCollator(tokenizer)

    train_samples, val_samples = load_jsonl(args.train_data), load_jsonl(args.val_data)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trainer = EpochTrainer(model, optimizer, CalibratedLoss(), device, accum_steps=accum_steps)
    best_val_loss = float("inf")
    print(f"[DATA] Train: {len(train_samples)} | Val: {len(val_samples)} | Steps/Epoch: {len(train_loader)}\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.perf_counter() - t0:.1f}s", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            with open(Path(args.output).with_suffix(".meta.json"), "w", encoding="utf-8") as f:
                json.dump({"checkpoint": args.output, "epoch": epoch, "val_loss": val_loss, "train_loss": train_loss, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)
            print(f"  -> Saved best checkpoint: {args.output} (val_loss: {val_loss:.4f})\n", flush=True)

if __name__ == "__main__":
    main()
