import argparse
from benchmark.dataset import (
    AbstentionSource,
    AgenticSource,
    IDataSource,
    LayaSource,
    TypedDecisionsSource,
)
from benchmark.runner import BenchmarkRunner
from benchmark.scorecard_layer import ScorecardLayer
from nanollm.inference.decision_engine import DEFAULT_CHECKPOINT

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

    benchmark = (
        BenchmarkRunner.from_checkpoint(args.checkpoint)
        | ScorecardLayer(track=args.suite, compare=args.compare)
    )
    benchmark.evaluate(target_suites, limit=args.limit)

if __name__ == "__main__":
    main()
