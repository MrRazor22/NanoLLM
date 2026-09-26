import argparse
from pathlib import Path
import torch
from transformers import AutoModel

from nanollm.inference.assembler_policy import SlotAssembler
from nanollm.model import ModelConfig, NanoModel
from nanollm.training import (
    CalibratedLoss,
    CheckpointingLayer,
    EpochTrainer,
    MultiQuestionCollator,
)
from training.dataset import TrainingDataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "train_adapt.jsonl"
DEFAULT_VAL = DEFAULT_DATA_DIR / "val_adapt.jsonl"
DEFAULT_OUTPUT = ROOT / "checkpoints" / "checkpoint_trained.pt"

def main() -> None:
    parser = argparse.ArgumentParser(description="Train NanoLLM")
    parser.add_argument("--train-data", type=str, default=str(DEFAULT_TRAIN), help="Path to training jsonl")
    parser.add_argument("--val-data", type=str, default=str(DEFAULT_VAL), help="Path to validation jsonl")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output checkpoint path")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Initial checkpoint path")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4000, help="Max tokens per batch for dynamic batching")
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--log-interval", type=float, default=10.0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assembler = SlotAssembler("answerdotai/ModernBERT-base")
    collator = MultiQuestionCollator(assembler)

    # 1. Training data policy: handles loading, extraction, and batch packing
    dataset = TrainingDataset(train_path=args.train_data, val_path=args.val_data)

    # 2. Model & Optimizer
    backbone = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
    config = ModelConfig(vocab_size=assembler.tokenizer.vocab_size, hidden_dim=768, num_layers=22, num_heads=12)
    model = NanoModel(config, backbone=backbone).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))

    pin = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=pin)
    loss_fn = CalibratedLoss()

    # 3. Training Engine wrapped with Checkpointing Layer
    trainer = EpochTrainer(
        model, optimizer, loss_fn, device, collator=collator, accum_steps=args.accum_steps, log_interval=args.log_interval
    ) | CheckpointingLayer(output_path=args.output)

    # 4. Run the full training session directly on the dataset policy
    print(f"Starting training on {device}...")
    trainer.fit(dataset, epochs=args.epochs, max_tokens=args.max_tokens, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
