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

    # 1. NanoLLM Champion
    print("\n--- Evaluating NanoLLM Champion ---", flush=True)
    nano_champ = DecisionEngine.from_checkpoint(str(ROOT / "checkpoints" / "checkpoint_champion.pt"))
    def decide_nano_champ(state, questions):
        qs = []
        for qid, spec in questions.items():
            t, ins, crit = spec["type"], spec["instructions"], spec["criteria"]
            if t == "choice":
                qs.append(Choice(qid, crit, instruction=ins))
            elif t == "noul":
                qs.append(Choice(qid, crit, instruction=ins))
            elif t == "score":
                opts = {str(i): c for i, c in enumerate(crit)} if isinstance(crit, list) else crit
                qs.append(Choice(qid, opts, instruction=ins))
        res = nano_champ.decide(state, qs)
        return {qid: res.answers[qid].choice for qid in questions}
    res_nano_champ = evaluate_model("NanoLLM Champion", decide_nano_champ, items)

    # 2. NanoLLM Clean Baseline
    print("\n--- Evaluating NanoLLM Clean Foundation ---", flush=True)
    nano_clean = DecisionEngine.from_checkpoint(str(ROOT / "checkpoints" / "checkpoint_clean.pt"))
    def decide_nano_clean(state, questions):
        qs = []
        for qid, spec in questions.items():
            t, ins, crit = spec["type"], spec["instructions"], spec["criteria"]
            if t == "choice":
                qs.append(Choice(qid, crit, instruction=ins))
            elif t == "noul":
                qs.append(Choice(qid, crit, instruction=ins))
            elif t == "score":
                opts = {str(i): c for i, c in enumerate(crit)} if isinstance(crit, list) else crit
                qs.append(Choice(qid, opts, instruction=ins))
        res = nano_clean.decide(state, qs)
        return {qid: res.answers[qid].choice for qid in questions}
    res_nano_clean = evaluate_model("NanoLLM Clean", decide_nano_clean, items)

    # 3. Laya Official SOTA
    print("\n--- Evaluating Laya Official SOTA ---", flush=True)
    laya_agent = laya.Agent("convaiinnovations/laya")
    def decide_laya(state, questions):
        out = laya_agent.predict(state, questions)
        preds = {}
        for qid, spec in questions.items():
            ans = out["answers"][qid]
            t = spec["type"]
            if t == "choice":
                preds[qid] = ans["choice"]
            elif t == "noul":
                preds[qid] = "true" if ans["noul"] >= 0.5 else "false"
            elif t == "score":
                preds[qid] = max(ans["probabilities"], key=ans["probabilities"].get)
        return preds
    res_laya = evaluate_model("Laya SOTA", decide_laya, items)

    # Print Side-by-Side Comparison Table
    print("\n" + "=" * 80)
    print(f"{'Category / Metric':30s} | {'NanoLLM Champ':14s} | {'NanoLLM Clean':14s} | {'Laya SOTA':14s}")
    print("-" * 80)
    cats = list(items[0]["category"] for items in [items]) # get order
    cat_keys = ["agent_tool_routing", "safety_guardrails", "triage_incident", "negative_constraints"]
    cat_labels = {
        "agent_tool_routing": "Agent Tool Routing (30)",
        "safety_guardrails": "Safety & Guardrails (30)",
        "triage_incident": "Incident Triage (90 Qs)",
        "negative_constraints": "Negative Constraints (30)"
    }
    for c in cat_keys:
        nc = res_nano_champ["by_cat"].get(c, 0.0) * 100
        nl = res_nano_clean["by_cat"].get(c, 0.0) * 100
        ly = res_laya["by_cat"].get(c, 0.0) * 100
        print(f"{cat_labels[c]:30s} | {nc:12.1f}% | {nl:12.1f}% | {ly:12.1f}%")
    print("-" * 80)
    print(f"{'OVERALL ACCURACY':30s} | {res_nano_champ['overall_acc']*100:12.1f}% | {res_nano_clean['overall_acc']*100:12.1f}% | {res_laya['overall_acc']*100:12.1f}%")
    print(f"{'Total Score':30s} | {res_nano_champ['total_correct']:5d}/{res_nano_champ['total_questions']:3d}      | {res_nano_clean['total_correct']:5d}/{res_nano_clean['total_questions']:3d}      | {res_laya['total_correct']:5d}/{res_laya['total_questions']:3d}")
    print(f"{'P50 Latency (CUDA)':30s} | {res_nano_champ['p50_ms']:10.1f} ms | {res_nano_clean['p50_ms']:10.1f} ms | {res_laya['p50_ms']:10.1f} ms")
    print(f"{'P90 Latency (CUDA)':30s} | {res_nano_champ['p90_ms']:10.1f} ms | {res_nano_clean['p90_ms']:10.1f} ms | {res_laya['p90_ms']:10.1f} ms")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
