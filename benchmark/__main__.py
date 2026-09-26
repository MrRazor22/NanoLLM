import argparse
from typing import Any
import torch
from benchmark.dataset import (
    AbstentionSource,
    AgenticSource,
    BenchmarkDataset,
    IDataSource,
    LayaSource,
    TypedDecisionsSource,
)
from benchmark.report import IScorecard, Scorecard
from nanollm.inference.decision_engine import DEFAULT_CHECKPOINT, DecisionEngine
from nanollm.inference.profiling_layer import ProfilingLayer

SUITE_SOURCES: dict[str, type[IDataSource]] = {
    "agentic": AgenticSource,
    "abstention": AbstentionSource,
    "typed_decisions": TypedDecisionsSource,
    "laya": LayaSource,
}

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
    scorecard: IScorecard = Scorecard()

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
        reports[key] = scorecard.evaluate_target(engine, ds, limit=args.limit)
        metadata[key] = meta

    scorecard.render(reports, track=args.suite, metadata=metadata, model_name="NanoLLM Champion")

if __name__ == "__main__":
    main()

