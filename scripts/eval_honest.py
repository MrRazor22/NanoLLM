import json, sys
from pathlib import Path

ROOT = Path("d:/CodeBase/NanoLLM")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "D:/CodeBase/laya")

from nanollm import Choice, DecisionEngine, ModelEvaluator, ProfilingEvaluator
import laya

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

def _decide_laya(agent, state, questions):
    out = agent.predict(state, questions)
    preds = {}
    for qid, spec in questions.items():
        ans = out["answers"][qid]
        t = spec["type"]
        if t == "choice": preds[qid] = ans["choice"]
        elif t == "noul": preds[qid] = "true" if ans["noul"] >= 0.5 else "false"
        elif t == "score": preds[qid] = max(ans["probabilities"], key=ans["probabilities"].get)
    return preds

def _print_table(r_v2, r_v1, r_ly):
    cats = {
        "agent_tool_routing": "Agent Tool Routing (30)",
        "safety_guardrails": "Safety & Guardrails (30)",
        "triage_incident": "Incident Triage (90 Qs)",
        "negative_constraints": "Negative Constraints (30)",
    }
    print("\n" + "=" * 80)
    print(f"{'Category / Metric':30s} | {'NanoLLM v2 (Tools)':18s} | {'NanoLLM v1':12s} | {'Laya SOTA':12s}")
    print("-" * 80)
    for c, label in cats.items():
        v2, v1, ly = r_v2["by_cat"].get(c, 0.0) * 100, r_v1["by_cat"].get(c, 0.0) * 100, r_ly["by_cat"].get(c, 0.0) * 100
        print(f"{label:30s} | {v2:16.1f}% | {v1:10.1f}% | {ly:10.1f}%")
    print("-" * 80)
    print(f"{'OVERALL ACCURACY':30s} | {r_v2['overall_acc']*100:16.1f}% | {r_v1['overall_acc']*100:10.1f}% | {r_ly['overall_acc']*100:10.1f}%")
    print(f"{'Total Score':30s} | {r_v2['total_correct']:5d}/{r_v2['total_questions']:3d}          | {r_v1['total_correct']:5d}/{r_v1['total_questions']:3d} | {r_ly['total_correct']:5d}/{r_ly['total_questions']:3d}")
    print(f"{'P50 Latency (CUDA)':30s} | {r_v2['p50_ms']:14.1f} ms | {r_v1['p50_ms']:8.1f} ms | {r_ly['p50_ms']:8.1f} ms")
    print(f"{'P90 Latency (CUDA)':30s} | {r_v2['p90_ms']:14.1f} ms | {r_v1['p90_ms']:8.1f} ms | {r_ly['p90_ms']:8.1f} ms")
    print("=" * 80 + "\n")

def main():
    bench_path = ROOT / "data" / "honest_benchmark.json"
    with open(bench_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} honest benchmark test items.")

    print("\n--- Evaluating NanoLLM v2 (With Tools) ---", flush=True)
    nano_v2 = DecisionEngine.from_checkpoint(str(ROOT / "checkpoints" / "checkpoint_champion_v2.pt"))
    r_v2 = ProfilingEvaluator(ModelEvaluator("NanoLLM v2", _make_nano_decide(nano_v2))).evaluate(items)

    print("\n--- Evaluating NanoLLM v1 ---", flush=True)
    nano_v1 = DecisionEngine.from_checkpoint(str(ROOT / "checkpoints" / "checkpoint_champion.pt"))
    r_v1 = ProfilingEvaluator(ModelEvaluator("NanoLLM v1", _make_nano_decide(nano_v1))).evaluate(items)

    print("\n--- Evaluating Laya Official SOTA ---", flush=True)
    laya_agent = laya.Agent("convaiinnovations/laya")
    r_ly = ProfilingEvaluator(ModelEvaluator("Laya SOTA", lambda s, q: _decide_laya(laya_agent, s, q))).evaluate(items)

    _print_table(r_v2, r_v1, r_ly)

if __name__ == "__main__":
    main()
