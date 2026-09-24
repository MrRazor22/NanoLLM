import json, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path("d:/CodeBase/NanoLLM")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "D:/CodeBase/laya")

from nanollm import DecisionEngine, Choice
import laya

def evaluate_model(name: str, decide_fn, items):
    stats = {}
    latencies = []
    
    # Warmup
    for _ in range(5):
        sample = items[0]
        decide_fn(sample["state"], sample["questions"])

    for item in items:
        cat = item["category"]
        if cat not in stats:
            stats[cat] = {"correct": 0, "total": 0}
        
        t0 = time.perf_counter()
        preds = decide_fn(item["state"], item["questions"])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        for qid, gold_spec in item["gold"].items():
            stats[cat]["total"] += 1
            pred_val = preds.get(qid)
            if pred_val == gold_spec["label"]:
                stats[cat]["correct"] += 1

    total_correct = sum(s["correct"] for s in stats.values())
    total_q = sum(s["total"] for s in stats.values())
    return {
        "name": name,
        "overall_acc": total_correct / total_q,
        "total_correct": total_correct,
        "total_questions": total_q,
        "by_cat": {c: s["correct"] / s["total"] for c, s in stats.items()},
        "p50_ms": float(np.median(latencies)),
        "p90_ms": float(np.percentile(latencies, 90))
    }

def main():
    bench_path = ROOT / "data" / "honest_benchmark.json"
    with open(bench_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} honest benchmark test items.")

    # 1. NanoLLM v2 (With 5k Real Tools)
    print("\n--- Evaluating NanoLLM v2 (With Tools) ---", flush=True)
    nano_v2 = DecisionEngine.from_checkpoint(str(ROOT / "checkpoints" / "checkpoint_champion_v2.pt"))
    def decide_nano_v2(state, questions):
        qs = []
        for qid, spec in questions.items():
            t, ins, crit = spec["type"], spec["instructions"], spec["criteria"]
            opts = {str(i): c for i, c in enumerate(crit)} if (t == "score" and isinstance(crit, list)) else crit
            qs.append(Choice(qid, opts, instruction=ins))
        res = nano_v2.decide(state, qs)
        return {qid: res.answers[qid].choice for qid in questions}
    res_v2 = evaluate_model("NanoLLM v2", decide_nano_v2, items)

    # 2. NanoLLM v1 (Champion)
    print("\n--- Evaluating NanoLLM v1 ---", flush=True)
    nano_v1 = DecisionEngine.from_checkpoint(str(ROOT / "checkpoints" / "checkpoint_champion.pt"))
    def decide_nano_v1(state, questions):
        qs = []
        for qid, spec in questions.items():
            t, ins, crit = spec["type"], spec["instructions"], spec["criteria"]
            opts = {str(i): c for i, c in enumerate(crit)} if (t == "score" and isinstance(crit, list)) else crit
            qs.append(Choice(qid, opts, instruction=ins))
        res = nano_v1.decide(state, qs)
        return {qid: res.answers[qid].choice for qid in questions}
    res_v1 = evaluate_model("NanoLLM v1", decide_nano_v1, items)

    # 3. Laya Official SOTA
    print("\n--- Evaluating Laya Official SOTA ---", flush=True)
    laya_agent = laya.Agent("convaiinnovations/laya")
    def decide_laya(state, questions):
        out = laya_agent.predict(state, questions)
        preds = {}
        for qid, spec in questions.items():
            ans = out["answers"][qid]
            t = spec["type"]
            if t == "choice": preds[qid] = ans["choice"]
            elif t == "noul": preds[qid] = "true" if ans["noul"] >= 0.5 else "false"
            elif t == "score": preds[qid] = max(ans["probabilities"], key=ans["probabilities"].get)
        return preds
    res_laya = evaluate_model("Laya SOTA", decide_laya, items)

    # Print Side-by-Side Comparison Table
    print("\n" + "=" * 80)
    print(f"{'Category / Metric':30s} | {'NanoLLM v2 (Tools)':18s} | {'NanoLLM v1':12s} | {'Laya SOTA':12s}")
    print("-" * 80)
    cat_keys = ["agent_tool_routing", "safety_guardrails", "triage_incident", "negative_constraints"]
    cat_labels = {
        "agent_tool_routing": "Agent Tool Routing (30)",
        "safety_guardrails": "Safety & Guardrails (30)",
        "triage_incident": "Incident Triage (90 Qs)",
        "negative_constraints": "Negative Constraints (30)"
    }
    for c in cat_keys:
        v2 = res_v2["by_cat"].get(c, 0.0) * 100
        v1 = res_v1["by_cat"].get(c, 0.0) * 100
        ly = res_laya["by_cat"].get(c, 0.0) * 100
        print(f"{cat_labels[c]:30s} | {v2:16.1f}% | {v1:10.1f}% | {ly:10.1f}%")
    print("-" * 80)
    print(f"{'OVERALL ACCURACY':30s} | {res_v2['overall_acc']*100:16.1f}% | {res_v1['overall_acc']*100:10.1f}% | {res_laya['overall_acc']*100:10.1f}%")
    print(f"{'Total Score':30s} | {res_v2['total_correct']:5d}/{res_v2['total_questions']:3d}          | {res_v1['total_correct']:5d}/{res_v1['total_questions']:3d} | {res_laya['total_correct']:5d}/{res_laya['total_questions']:3d}")
    print(f"{'P50 Latency (CUDA)':30s} | {res_v2['p50_ms']:14.1f} ms | {res_v1['p50_ms']:8.1f} ms | {res_laya['p50_ms']:8.1f} ms")
    print(f"{'P90 Latency (CUDA)':30s} | {res_v2['p90_ms']:14.1f} ms | {res_v1['p90_ms']:8.1f} ms | {res_laya['p90_ms']:8.1f} ms")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
