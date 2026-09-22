import os
import sys
import time
from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from nanollm import (
    DecisionSample,
    ModelConfig,
    MultiQuestionCollator,
    NanoModel,
    SubwordTokenizer,
)
from nanollm.dataset import load_jsonl

DEPARTMENTS = ["Billing", "Infrastructure / Tech", "Enterprise / Sales", "Security"]

def evaluate_dataset(model: NanoModel, collator: MultiQuestionCollator, device: torch.device, path: str):
    if not os.path.exists(path):
        print(f"Dataset path does not exist: {path}")
        return

    samples = load_jsonl(path)
    loader = DataLoader(samples, batch_size=32, shuffle=False, collate_fn=collator)

    total_samples = 0
    choice_correct = 0
    noul_correct = 0
    noul_brier_sum = 0.0
    score_mae_sum = 0.0

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            positions = batch["question_positions"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            logits = model(input_ids, positions, mask)
            b_size = input_ids.shape[0]
            total_samples += b_size

            choice_preds = logits[:, 0, :len(DEPARTMENTS)].argmax(dim=-1)
            choice_correct += (choice_preds == targets[:, 0].long()).sum().item()

            noul_probs = torch.sigmoid(logits[:, 1, 0])
            noul_binary = (noul_probs >= 0.5).float()
            noul_correct += (noul_binary == targets[:, 1]).sum().item()
            noul_brier_sum += ((noul_probs - targets[:, 1]) ** 2).sum().item()

            score_preds = torch.sigmoid(logits[:, 2, 0]) * 100.0
            score_targets = targets[:, 2] * 100.0
            score_mae_sum += (score_preds - score_targets).abs().sum().item()

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    avg_latency = elapsed_ms / max(1, total_samples)

    print("==================================================")
    print(f"          STANDARDIZED EVALUATION REPORT          ")
    print("==================================================")
    print(f"Target Dataset:       {path} ({total_samples} samples)")
    print(f"Inference Latency:    {avg_latency:.2f} ms / sample")
    print("--------------------------------------------------")
    print(f"Choice Accuracy:      {(choice_correct / total_samples) * 100:.2f}%")
    print(f"Noul Accuracy:        {(noul_correct / total_samples) * 100:.2f}%")
    print(f"Noul Brier Score:     {noul_brier_sum / total_samples:.4f} (0.0 = perfect)")
    print(f"Score Error (MAE):    {score_mae_sum / total_samples:.2f} points")
    print("==================================================")

def predict_single(model: NanoModel, collator: MultiQuestionCollator, device: torch.device, query: str):
    sample = DecisionSample(state=query, questions=[("dept", "choice", 0), ("urgent", "noul", 0.0), ("score", "score", 0.0)])
    batch = collator([sample])
    with torch.no_grad():
        logits = model(batch["input_ids"].to(device), batch["question_positions"].to(device), batch["mask"].to(device))

    probs = logits[0, 0, :len(DEPARTMENTS)].softmax(dim=-1)
    best_idx = int(probs.argmax().item())
    urgent_prob = torch.sigmoid(logits[0, 1, 0]).item()
    severity = torch.sigmoid(logits[0, 2, 0]).item() * 100.0

    print(f"\nQuery: {query}")
    print(f"  Department: {DEPARTMENTS[best_idx]} ({probs[best_idx] * 100:.1f}%)")
    print(f"  Urgent:     {urgent_prob > 0.5} (p={urgent_prob:.3f})")
    print(f"  Severity:   {severity:.1f}/100")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SubwordTokenizer("bert-base-uncased")
    collator = MultiQuestionCollator(tokenizer)

    backbone = AutoModel.from_pretrained("bert-base-uncased")
    config = ModelConfig(vocab_size=tokenizer.vocab_size, hidden_dim=768, num_layers=12, num_heads=12, max_choices=16)
    model = NanoModel(config, backbone=backbone).to(device)
    if os.path.exists("checkpoint.pt"):
        model.load_state_dict(torch.load("checkpoint.pt", map_location=device))
    model.eval()

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        predict_single(model, collator, device, " ".join(sys.argv[1:]))
    else:
        dataset_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--data" else "data/test.jsonl"
        evaluate_dataset(model, collator, device, dataset_path)

if __name__ == "__main__":
    main()
