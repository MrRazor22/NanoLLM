from pathlib import Path
from typing import Any, Dict, Optional
import argparse, json, sys, time
import numpy as np
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from nanollm import Choice, DecisionEngine, HierarchicalLayer, ModelEvaluator, ProfilingEvaluator
from scripts.taxonomies import BANKING_CLUSTERS, MASSIVE_CLUSTERS

BENCHMARK_TARGETS = {
    "ag_news": ("AG News", "fancyzhx/ag_news", "test", "text", "label", {
        "World": "international news, politics, conflicts",
        "Sports": "sports, games, athletes",
        "Business": "companies, markets, economy",
        "Sci/Tech": "science, technology, software, space"
    }, None, 0.953, "What is the topic of the article?"),
    "emotion": ("DAIR Emotion", "dair-ai/emotion", "test", "text", "label", ["sadness", "joy", "love", "anger", "fear", "surprise"], None, 0.600, "Which emotion is most strongly expressed in this text?"),
    "massive": ("MASSIVE Intent", "mteb/amazon_massive_intent", "test", "text", "label_text", None, "en", 0.783, "What is the user's intent in this utterance?", MASSIVE_CLUSTERS),
    "banking77": ("Banking77", "mteb/banking77", "test", "text", "label_text", None, None, 0.492, "Which banking intent does this message express?", BANKING_CLUSTERS),
}

def eval_typed_decisions(engine: DecisionEngine, max_n: Optional[int]) -> Dict[str, Any]:
    print("\n[BENCHMARK] Evaluating LocalLLaMA/typed-decisions...", flush=True)
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    if max_n: ds = ds.select(range(min(len(ds), max_n)))
    items = []
    for item in ds:
        q_dict, g_dict = json.loads(item["questions"]), json.loads(item["gold"])
        qs, gold = {}, {}
        for name, spec in q_dict.items():
            t, ins, crit = spec.get("type"), spec.get("instructions"), spec.get("criteria", {})
            opts = crit if isinstance(crit, dict) else list(crit)
            if t == "noul" and not (isinstance(crit, dict) and crit):
                opts = {"false": "no, condition does not hold", "true": "yes, condition holds"}
            elif t == "score" and isinstance(crit, list):
                opts = {str(i): c for i, c in enumerate(crit)}
            qs[name] = {"type": "choice", "instructions": ins, "criteria": opts}
            g_lbl = g_dict.get(name, {}).get("label")
            if t == "noul": g_lbl = "true" if (g_lbl == "true" or g_dict.get(name, {}).get("noul", 0) >= 0.5) else "false"
            gold[name] = {"label": str(g_lbl)}
        items.append({"state": item["state"], "questions": qs, "gold": gold})

    def decide(state, questions):
        clist = [Choice(qid, sp["criteria"], instruction=sp["instructions"]) for qid, sp in questions.items()]
        res = engine.decide(state, clist)
        return {qid: res.answers[qid].choice for qid in questions}

    r = ProfilingEvaluator(ModelEvaluator("typed-decisions", decide)).evaluate(items)
    return {"name": "typed-decisions", "acc": r["overall_acc"], "laya_acc": 0.766, "p50": r.get("p50_ms", 0.0)}

def eval_choice_dataset(engine: Any, cfg: tuple, max_n: Optional[int]) -> Dict[str, Any]:
    name, path, split, tcol, lcol, opts, sub, laya_tgt, instr = cfg[:9]
    clusters = cfg[9] if len(cfg) > 9 else None
    print(f"\n[BENCHMARK] Evaluating {name}...", flush=True)
    ds = load_dataset(path, sub, split=split) if sub else load_dataset(path, split=split)
    if not opts:
        train_ds = load_dataset(path, sub, split="train") if sub else load_dataset(path, split="train")
        opts = sorted(list(set(train_ds[lcol])))
    if max_n: ds = ds.shuffle(seed=42).select(range(min(len(ds), max_n)))

    items = []
    opt_keys = list(opts.keys()) if isinstance(opts, dict) else opts
    for row in ds:
        g = row[lcol]
        target = opt_keys[g] if isinstance(g, int) else str(g)
        items.append({
            "state": str(row[tcol]).strip(),
            "questions": {"label": {"type": "choice", "instructions": instr, "criteria": opts}},
            "gold": {"label": {"label": target}}
        })

    def decide(state, questions):
        res = engine.decide(state, [Choice("label", opts, instruction=instr, clusters=clusters)])
        return {"label": res.answers["label"].choice}

    r = ProfilingEvaluator(ModelEvaluator(name, decide)).evaluate(items)
    return {"name": name, "acc": r["overall_acc"], "laya_acc": laya_tgt, "p50": r.get("p50_ms", 0.0)}

def main():
    parser = argparse.ArgumentParser(description="Official NanoLLM vs Laya Scientific Benchmark Suite")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_champion.pt")
    parser.add_argument("--samples", type=int, default=500, help="Samples per dataset (0 = all)")
    parser.add_argument("--task", default="all", choices=["all", "typed_decisions", "massive", "banking77", "ag_news", "emotion"])
    parser.add_argument("--hierarchical", action="store_true", help="Wrap engine in HierarchicalLayer")
    args = parser.parse_args()

    raw_engine = DecisionEngine.from_checkpoint(args.checkpoint)
    engine = HierarchicalLayer(raw_engine) if args.hierarchical else raw_engine
    n = None if args.samples == 0 else args.samples
    results = []

    if args.task in ["all", "typed_decisions"]: results.append(eval_typed_decisions(engine, n))
    for k, cfg in BENCHMARK_TARGETS.items():
        if args.task in ["all", k]: results.append(eval_choice_dataset(engine, cfg, n))

    print("\n" + "=" * 65)
    print(f"{'Benchmark Dataset':25s} | {'NanoLLM':10s} | {'Laya':10s} | {'P50 (ms)':8s}")
    print("-" * 65)
    for r in results:
        laya_str = f"{r['laya_acc'] * 100:.1f}%" if r["laya_acc"] else "N/A"
        print(f"{r['name']:25s} | {r['acc'] * 100:8.1f}% | {laya_str:10s} | {r['p50']:6.1f}ms")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
