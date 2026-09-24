from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse, json, sys, time
import numpy as np, torch
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from nanollm import Choice, DecisionEngine, HierarchicalLayer, Noul, Score
from scripts.taxonomies import BANKING_CLUSTERS, MASSIVE_CLUSTERS

BENCHMARK_TARGETS = {
    "ag_news": ("AG News", "fancyzhx/ag_news", "test", "text", "label", {
        "World": "international news, politics, conflicts",
        "Sports": "sports, games, athletes",
        "Business": "companies, markets, economy",
        "Sci/Tech": "science, technology, software, space"
    }, None, 0.953, None),
    "emotion": ("DAIR Emotion", "dair-ai/emotion", "test", "text", "label", ["sadness", "joy", "love", "anger", "fear", "surprise"], None, 0.600, "Which emotion is most strongly expressed in this text?"),
    "massive": ("MASSIVE Intent", "mteb/amazon_massive_intent", "test", "text", "label_text", None, "en", 0.783, "What is the user's intent in this utterance?", MASSIVE_CLUSTERS),
    "banking77": ("Banking77", "mteb/banking77", "test", "text", "label_text", None, None, 0.492, "Which banking intent does this message express?", BANKING_CLUSTERS),
}


def eval_typed_decisions(engine: DecisionEngine, max_n: Optional[int]) -> Dict[str, Any]:
    print("\n[BENCHMARK] Evaluating LocalLLaMA/typed-decisions...", flush=True)
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    if max_n: ds = ds.select(range(min(len(ds), max_n)))
    correct, total, latencies = 0, 0, []

    for idx, item in enumerate(ds):
        q_dict, gold_dict = json.loads(item["questions"]), json.loads(item["gold"])
        questions = []
        for name, spec in q_dict.items():
            t, ins, crit = spec.get("type"), spec.get("instructions"), spec.get("criteria", {})
            if t == "choice":
                questions.append(Choice(name, crit if isinstance(crit, dict) else list(crit), instruction=ins))
            elif t == "noul":
                questions.append(Noul(name, instruction=ins))
            elif t == "score":
                opts = {str(i): c for i, c in enumerate(crit)} if isinstance(crit, list) else crit
                questions.append(Choice(name, opts, instruction=ins))

        res = engine.decide(item["state"], questions)
        latencies.append(res.latency_ms)

        for name, gold in gold_dict.items():
            ans = res.answers.get(name)
            if not ans: continue
            total += 1
            if gold.get("type") in ("choice", "score") and ans.choice == gold.get("label"): correct += 1
            elif gold.get("type") == "noul" and ans.value == (gold.get("label") == "true" or gold.get("noul", 0) >= 0.5): correct += 1


        if (idx + 1) % 100 == 0 or (idx + 1) == len(ds):
            print(f"  typed-decisions [{idx + 1:4d}/{len(ds)}] Running Acc: {(correct / max(1, total)) * 100:5.1f}%", flush=True)

    return {"name": "typed-decisions", "acc": correct / max(1, total), "laya_acc": 0.766, "p50": float(np.median(latencies))}

def eval_choice_dataset(engine: Any, cfg: tuple, max_n: Optional[int]) -> Dict[str, Any]:
    name, path, split, tcol, lcol, opts, sub, laya_tgt, instr = cfg[:9]
    clusters = cfg[9] if len(cfg) > 9 else None
    print(f"\n[BENCHMARK] Evaluating {name}...", flush=True)
    ds = load_dataset(path, sub, split=split) if sub else load_dataset(path, split=split)
    if not opts:
        train_ds = load_dataset(path, sub, split="train") if sub else load_dataset(path, split="train")
        opts = sorted(list(set(train_ds[lcol])))
    if max_n: ds = ds.shuffle(seed=42).select(range(min(len(ds), max_n)))

    correct, latencies = 0, []
    opt_keys = list(opts.keys()) if isinstance(opts, dict) else opts
    for idx, item in enumerate(ds):
        gold = item[lcol]
        target = opt_keys[gold] if isinstance(gold, int) else str(gold)
        res = engine.decide(str(item[tcol]).strip(), [Choice("label", opts, instruction=instr, clusters=clusters)])
        latencies.append(res.latency_ms)
        if res.answers["label"].choice == target: correct += 1
        if (idx + 1) % 100 == 0 or (idx + 1) == len(ds):
            print(f"  {name} [{idx + 1:4d}/{len(ds)}] Running Acc: {(correct / (idx + 1)) * 100:5.1f}%", flush=True)


    return {"name": name, "acc": correct / max(1, len(ds)), "laya_acc": laya_tgt, "p50": float(np.median(latencies))}

def main():
    parser = argparse.ArgumentParser(description="Official NanoLLM vs Laya Scientific Benchmark Suite")
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--samples", type=int, default=500, help="Samples per dataset (0 = all)")
    parser.add_argument("--task", default="all", choices=["all", "typed_decisions", "massive", "banking77", "ag_news", "emotion"])
    args = parser.parse_args()

    engine = HierarchicalLayer(DecisionEngine.from_checkpoint(args.checkpoint))
    n = None if args.samples == 0 else args.samples
    results = []

    if args.task in ["all", "typed_decisions"]:
        results.append(eval_typed_decisions(engine, n))

    for k, cfg in BENCHMARK_TARGETS.items():
        if args.task in ["all", k]:
            results.append(eval_choice_dataset(engine, cfg, n))

    print("\n" + "=" * 65)
    print(f"{'Benchmark Dataset':25s} | {'NanoLLM':10s} | {'Laya':10s} | {'P50 (ms)':8s}")
    print("-" * 65)
    for r in results:
        laya_str = f"{r['laya_acc'] * 100:.1f}%" if r["laya_acc"] else "N/A"
        print(f"{r['name']:25s} | {r['acc'] * 100:8.1f}% | {laya_str:10s} | {r['p50']:6.1f}ms")
    print("=" * 65 + "\n")

    res_dir = ROOT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = res_dir / "benchmarks.json"
    history = []
    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f: history = json.load(f)
        except Exception: history = []
    history.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": args.checkpoint,
        "samples": args.samples,
        "results": results
    })
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[LEDGER] Appended benchmark results to {ledger_path}", flush=True)


if __name__ == "__main__":
    main()
