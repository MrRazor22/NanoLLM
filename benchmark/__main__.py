import argparse
import torch

from nanollm.inference.engine import DEFAULT_CHECKPOINT, DecisionEngine
from benchmark.dataset import (
    AbstentionPolicy,
    AgenticPolicy,
    BenchmarkDataset,
    ISuiteSourcePolicy,
    LayaPolicy,
    TypedDecisionsPolicy,
)

SUITE_POLICIES: dict[str, type[ISuiteSourcePolicy]] = {
    "agentic": AgenticPolicy,
    "abstention": AbstentionPolicy,
    "typed_decisions": TypedDecisionsPolicy,
    "laya": LayaPolicy,
}

def main() -> None:
    suites = SUITE_POLICIES
    parser = argparse.ArgumentParser(description="NanoLLM Honest Benchmark")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to checkpoint")
    parser.add_argument("--suite", "--track", type=str, default="all", dest="suite", choices=["all"] + list(suites.keys()), help="Benchmark suite to run")
    parser.add_argument("--all", action="store_const", dest="suite", const="all", help="Run all suites")
    for name in suites:
        parser.add_argument(f"--{name.replace('_', '-')}", action="store_const", dest="suite", const=name, help=f"Run {name} suite")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for quick validation")
    parser.add_argument("--log-interval", type=int, default=100, help="Live step logging interval")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = DecisionEngine.from_checkpoint(args.checkpoint, device=device)
    evaluator = ModelEvaluator.from_engine("NanoLLM Champion", engine, log_interval=args.log_interval) | ProfilingEvaluatorLayer(warmup_runs=5)

    if args.suite == "all":
        reports = {}
        for name, suite_cls in suites.items():
            print(f"\n{'='*70}\n>>> Running Benchmark Suite: {name.upper().replace('_', ' ')}\n{'='*70}", flush=True)
            reports[name] = evaluator.evaluate(BenchmarkDataset(suite_cls()), limit=args.limit)
        print_consolidated_scorecard(reports)
    else:
        suite_cls = suites.get(args.suite)
        if suite_cls is None:
            raise ValueError(f"Unknown benchmark suite: '{args.suite}'. Available: {list(suites.keys())}")
        report = evaluator.evaluate(BenchmarkDataset(suite_cls()), limit=args.limit)
        print_scorecard(report, track=args.suite)

if __name__ == "__main__":
    main()
