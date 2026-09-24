import argparse, json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanollm import (
    AdaptationCurriculum,
    CalibratedLoss,
    CheckpointingLayer,
    DecisionEngine,
    EpochTrainer,
    ModelConfig,
    ModelEvaluator,
    MultiQuestionCollator,
    NanoModel,
    ProfilingEvaluatorLayer,
    SubwordTokenizer,
    load_jsonl,
    print_benchmark_table,
    save_jsonl,
)

def run_eval(benchmark_file: str):
    with open(benchmark_file, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} honest benchmark test items.")

    print("\n--- Evaluating NanoLLM v2 (With Tools) ---", flush=True)
    nano_v2 = DecisionEngine.from_checkpoint("checkpoints/checkpoint_champion_v2.pt")
    r_v2 = ProfilingEvaluatorLayer(ModelEvaluator.from_engine("NanoLLM v2", nano_v2)).evaluate(items)

    print("\n--- Evaluating NanoLLM v1 ---", flush=True)
    nano_v1 = DecisionEngine.from_checkpoint("checkpoints/checkpoint_champion.pt")
    r_v1 = ProfilingEvaluatorLayer(ModelEvaluator.from_engine("NanoLLM v1", nano_v1)).evaluate(items)

    r_ly = {"name": "Laya SOTA", "by_cat": {}, "overall_acc": 0.0, "total_correct": 0, "total_questions": len(items)}
    try:
        if Path("D:/CodeBase/laya").exists(): sys.path.insert(0, "D:/CodeBase/laya")
        import laya
        print("\n--- Evaluating Laya Official SOTA ---", flush=True)
        laya_agent = laya.Agent("convaiinnovations/laya")
        def decide_laya(s, q):
            out = laya_agent.predict(s, q)
            return {qid: out["answers"][qid]["choice"] if spec["type"] == "choice" else ("true" if out["answers"][qid]["noul"] >= 0.5 else "false") if spec["type"] == "noul" else max(out["answers"][qid]["probabilities"], key=out["answers"][qid]["probabilities"].get) for qid, spec in q.items()}
        r_ly = ProfilingEvaluatorLayer(ModelEvaluator("Laya SOTA", decide_laya)).evaluate(items)
    except Exception:
        pass

    print_benchmark_table([r_v2, r_v1, r_ly])

def run_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SubwordTokenizer("answerdotai/ModernBERT-base")
    collator = MultiQuestionCollator(tok)
    train_s, val_s = load_jsonl(args.train_data), load_jsonl(args.val_data)
    if args.max_samples > 0:
        train_s, val_s = train_s[:args.max_samples], val_s[:max(100, args.max_samples // 5)]
    train_loader = DataLoader(train_s, batch_size=8, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_s, batch_size=8, shuffle=False, collate_fn=collator)
    bb = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
    cfg = ModelConfig(vocab_size=tok.vocab_size, hidden_dim=768, num_layers=22, num_heads=12)
    model = NanoModel(cfg, backbone=bb).to(device)
    if args.init and Path(args.init).exists():
        model.load_state_dict(torch.load(args.init, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trainer = CheckpointingLayer(EpochTrainer(model, opt, CalibratedLoss(), device, accum_steps=4), args.output, val_loader)
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch} | Train Loss: {loss:.4f} | Best Val: {trainer.best_val_loss:.4f} | Time: {time.perf_counter() - t0:.1f}s")

def run_adapt(args):
    curriculum = AdaptationCurriculum(args.data_dir)
    train_data, val_data = curriculum.build()
    save_jsonl(str(Path(args.data_dir) / "train_adapt.jsonl"), train_data)
    save_jsonl(str(Path(args.data_dir) / "val_adapt.jsonl"), val_data)
    print(f"Adapted dataset generated: {len(train_data)} train, {len(val_data)} val.")

def main():
    p = argparse.ArgumentParser(description="NanoLLM Unified CLI")
    sub = p.add_subparsers(dest="command", required=True)
    pe = sub.add_parser("eval", help="Run honest evaluation benchmark")
    pe.add_argument("--benchmark", default="data/honest_benchmark.json")
    pt = sub.add_parser("train", help="Train NanoLLM model")
    pt.add_argument("--output", default="checkpoints/checkpoint.pt")
    pt.add_argument("--init", default=None)
    pt.add_argument("--epochs", type=int, default=2)
    pt.add_argument("--lr", type=float, default=2e-5)
    pt.add_argument("--train-data", default="data/train_adapt.jsonl")
    pt.add_argument("--val-data", default="data/val_adapt.jsonl")
    pt.add_argument("--max-samples", type=int, default=0)
    pa = sub.add_parser("adapt", help="Generate adaptation curriculum dataset")
    pa.add_argument("--data-dir", default="data")
    args = p.parse_args()
    if args.command == "eval": run_eval(args.benchmark)
    elif args.command == "train": run_train(args)
    elif args.command == "adapt": run_adapt(args)

if __name__ == "__main__":
    main()
