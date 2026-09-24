import argparse, json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

if Path("D:/CodeBase/laya").exists():
    sys.path.insert(0, "D:/CodeBase/laya")

from nanollm import (
    AdaptationCurriculum,
    CalibratedLoss,
    CheckpointingLayer,
    Choice,
    DecisionEngine,
    EpochTrainer,
    ModelConfig,
    ModelEvaluator,
    MultiQuestionCollator,
    NanoModel,
    ProfilingEvaluatorLayer,
    SubwordTokenizer,
    load_jsonl,
    save_jsonl,
)

def _make_nano_decide(engine: DecisionEngine):
    def decide(state, questions):
        qs = []
        for qid, spec in questions.items():
            t, ins, crit = spec["type"], spec["instructions"], spec["criteria"]
            opts = {str(i): c for i, c in enumerate(crit)} if (t == "score" and isinstance(crit, list)) else crit
            qs.append(Choice(qid, opts, instruction=ins))
        res = engine.decide(state, qs)
        return {qid: res.answers[qid].choice for qid in questions}
    return decide

def _eval_honest(benchmark_file: str):
    with open(benchmark_file, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} honest benchmark test items.")

    print("\n--- Evaluating NanoLLM v2 (With Tools) ---", flush=True)
    nano_v2 = DecisionEngine.from_checkpoint("checkpoints/checkpoint_champion_v2.pt")
    r_v2 = ProfilingEvaluatorLayer(ModelEvaluator("NanoLLM v2", _make_nano_decide(nano_v2))).evaluate(items)

    print("\n--- Evaluating NanoLLM v1 ---", flush=True)
    nano_v1 = DecisionEngine.from_checkpoint("checkpoints/checkpoint_champion.pt")
    r_v1 = ProfilingEvaluatorLayer(ModelEvaluator("NanoLLM v1", _make_nano_decide(nano_v1))).evaluate(items)

    try:
        import laya
        print("\n--- Evaluating Laya Official SOTA ---", flush=True)
        laya_agent = laya.Agent("convaiinnovations/laya")
        def decide_laya(s, q):
            out = laya_agent.predict(s, q)
            return {qid: out["answers"][qid]["choice"] if spec["type"] == "choice" else ("true" if out["answers"][qid]["noul"] >= 0.5 else "false") if spec["type"] == "noul" else max(out["answers"][qid]["probabilities"], key=out["answers"][qid]["probabilities"].get) for qid, spec in q.items()}
        r_ly = ProfilingEvaluatorLayer(ModelEvaluator("Laya SOTA", decide_laya)).evaluate(items)
    except Exception:
        r_ly = {"by_cat": {}, "overall_acc": 0.0, "total_correct": 0, "total_questions": len(items), "p50_ms": 0.0, "p90_ms": 0.0}

    cats = {"agent_tool_routing": "Agent Tool Routing (30)", "safety_guardrails": "Safety & Guardrails (30)", "triage_incident": "Incident Triage (90 Qs)", "negative_constraints": "Negative Constraints (30)"}
    print("\n" + "=" * 80)
    print(f"{'Category / Metric':30s} | {'NanoLLM v2 (Tools)':18s} | {'NanoLLM v1':12s} | {'Laya SOTA':12s}")
    print("-" * 80)
    for c, label in cats.items():
        print(f"{label:30s} | {r_v2['by_cat'].get(c, 0)*100:16.1f}% | {r_v1['by_cat'].get(c, 0)*100:10.1f}% | {r_ly['by_cat'].get(c, 0)*100:10.1f}%")
    print("-" * 80)
    print(f"{'OVERALL ACCURACY':30s} | {r_v2['overall_acc']*100:16.1f}% | {r_v1['overall_acc']*100:10.1f}% | {r_ly['overall_acc']*100:10.1f}%")
    print(f"{'Total Score':30s} | {r_v2['total_correct']:5d}/{r_v2['total_questions']:3d}          | {r_v1['total_correct']:5d}/{r_v1['total_questions']:3d} | {r_ly['total_correct']:5d}/{r_ly['total_questions']:3d}")
    print(f"{'P50 Latency (CUDA)':30s} | {r_v2.get('p50_ms', 0):14.1f} ms | {r_v1.get('p50_ms', 0):8.1f} ms | {r_ly.get('p50_ms', 0):8.1f} ms")
    print(f"{'P90 Latency (CUDA)':30s} | {r_v2.get('p90_ms', 0):14.1f} ms | {r_v1.get('p90_ms', 0):8.1f} ms | {r_ly.get('p90_ms', 0):8.1f} ms")
    print("=" * 80 + "\n")

def _train(args):
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
    raw_trainer = EpochTrainer(model, opt, CalibratedLoss(), device, accum_steps=4)
    trainer = CheckpointingLayer(raw_trainer, args.output, val_loader)
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch} | Train Loss: {loss:.4f} | Best Val: {trainer.best_val_loss:.4f} | Time: {time.perf_counter() - t0:.1f}s")

def _adapt(args):
    curriculum = AdaptationCurriculum(args.data_dir)
    train_data, val_data = curriculum.build()
    save_jsonl(train_data, str(Path(args.data_dir) / "train_adapt.jsonl"))
    save_jsonl(val_data, str(Path(args.data_dir) / "val_adapt.jsonl"))
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
    if args.command == "eval": _eval_honest(args.benchmark)
    elif args.command == "train": _train(args)
    elif args.command == "adapt": _adapt(args)

if __name__ == "__main__":
    main()
