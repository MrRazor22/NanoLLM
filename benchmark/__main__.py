import argparse
from typing import Any
import numpy as np
import torch
from benchmark.competitor import CompetitorScorecard
from benchmark.dataset import (
    AbstentionSource,
    AgenticSource,
    BenchmarkDataset,
    IDataSource,
    LayaSource,
    TypedDecisionsSource,
)
from nanollm.inference.decision_engine import DEFAULT_CHECKPOINT, DecisionEngine
from nanollm.inference.profiling_layer import ProfilingLayer
from nanollm.inference.schema import Choice

SUITE_SOURCES: dict[str, type[IDataSource]] = {
    "agentic": AgenticSource,
    "abstention": AbstentionSource,
    "typed_decisions": TypedDecisionsSource,
    "laya": LayaSource,
}

def evaluate_target(target: Any, dataset: Any, limit: int | None = None) -> dict[str, Any]:
    items = dataset.load(limit=limit) if hasattr(dataset, "load") else (dataset[:limit] if limit else dataset)
    stats: dict[str, dict[str, int]] = {}
    latencies: list[float] = []

    for item in items:
        cat = item.get("category", "general")
        if cat not in stats:
            stats[cat] = {"correct": 0, "total": 0}

        if hasattr(target, "decide"):
            qs = [
                Choice(qid, spec.get("options") or spec.get("criteria", {}), instruction=spec.get("instructions"))
                for qid, spec in item["questions"].items()
            ]
            res = target.decide(item["state"], qs)
            preds = {qid: str(ans.choice) for qid, ans in res.answers.items()}
            if hasattr(res, "latency_ms") and res.latency_ms is not None:
                latencies.append(res.latency_ms)
        else:
            raw = target(item["state"], item["questions"])
            preds = {qid: (val["choice"] if isinstance(val, dict) else str(val)) for qid, val in raw.items()}

        for qid, gold in item["gold"].items():
            if qid in preds:
                stats[cat]["total"] += 1
                if preds[qid] == str(gold.get("label", "")):
                    stats[cat]["correct"] += 1

    total_correct = sum(s["correct"] for s in stats.values())
    total_eval = sum(s["total"] for s in stats.values())
    report: dict[str, Any] = {
        "overall_acc": (total_correct / total_eval) if total_eval > 0 else 0.0,
        "total_correct": total_correct,
        "total_questions": total_eval,
        "by_cat": {c: (s["correct"] / s["total"]) if s["total"] > 0 else 0.0 for c, s in stats.items()},
        "cat_counts": {c: s["total"] for c, s in stats.items()},
    }
    if latencies:
        report["p50_ms"] = float(np.median(latencies))
    return report

def main() -> None:
    suites = SUITE_SOURCES
    parser = argparse.ArgumentParser(description="NanoLLM Honest Benchmark")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to checkpoint")
    parser.add_argument("--suite", "--track", type=str, default="all", dest="suite", choices=["all"] + list(suites.keys()), help="Benchmark suite to run")
    parser.add_argument("--all", action="store_const", dest="suite", const="all", help="Run all suites")
    for name in suites:
        parser.add_argument(f"--{name.replace('_', '-')}", action="store_const", dest="suite", const=name, help=f"Run {name} suite")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for quick validation")
    parser.add_argument("--compare", action="store_true", help="Render competitive benchmark matrix against baselines")
    args = parser.parse_args()

    target_suites = suites if args.suite == "all" else {args.suite: suites[args.suite]}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    engine = DecisionEngine.from_checkpoint(args.checkpoint, device=dev) | ProfilingLayer()

    reports: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for key, src_cls in target_suites.items():
        ds = BenchmarkDataset(src_cls())
        meta = {
            "display_name": getattr(ds, "display_name", key.replace("_", " ").title()),
            "category_labels": getattr(ds, "category_labels", {}),
        }
        if len(target_suites) > 1:
            print(f"\n{'='*70}\n>>> Running Benchmark Suite: {meta['display_name'].upper()}\n{'='*70}", flush=True)
        reports[key] = evaluate_target(engine, ds, limit=args.limit)
        metadata[key] = meta

    CompetitorScorecard().render(reports, track=args.suite, metadata=metadata, model_name="NanoLLM Champion")

if __name__ == "__main__":
    main()

