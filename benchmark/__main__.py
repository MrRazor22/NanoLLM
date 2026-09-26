import argparse
import torch

from nanollm.inference.engine import DEFAULT_CHECKPOINT, DecisionEngine
from nanollm.inference.profiling_layer import ProfilingLayer
from benchmark.dataset import (
    AbstentionSource,
    AgenticSource,
    BenchmarkDataset,
    IDataSource,
    LayaSource,
    TypedDecisionsSource,
)
from benchmark.competitor import CompetitorScorecard
from benchmark.evaluator import ModelEvaluator
from benchmark.reporting_layer import ReportingEvaluatorLayer

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
    parser.add_argument("--log-interval", type=int, default=100, help="Live step logging interval")
    parser.add_argument("--compare", action="store_true", help="Render competitive benchmark matrix against baselines")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Profile inference latency directly through the inference engine layer
    engine = DecisionEngine.from_checkpoint(args.checkpoint, device=device) | ProfilingLayer()
    base_evaluator = ModelEvaluator.from_engine("NanoLLM Champion", engine, log_interval=args.log_interval)

    if args.suite == "all":
        reports = {}
        metadata = {}
        for name, suite_cls in suites.items():
            print(f"\n{'='*70}\n>>> Running Benchmark Suite: {name.upper().replace('_', ' ')}\n{'='*70}", flush=True)
            ds = BenchmarkDataset(suite_cls())
            metadata[name] = {
                "display_name": ds.display_name,
                "category_labels": ds.category_labels,
            }
            reports[name] = base_evaluator.evaluate(ds, limit=args.limit)

        CompetitorScorecard().render(reports, track="all", metadata=metadata, model_name="NanoLLM Champion")
    else:
        suite_cls = suites.get(args.suite)
        if suite_cls is None:
            raise ValueError(f"Unknown benchmark suite: '{args.suite}'. Available: {list(suites.keys())}")
        evaluator = base_evaluator | ReportingEvaluatorLayer(track=args.suite)
        report = evaluator.evaluate(BenchmarkDataset(suite_cls()), limit=args.limit)
        if args.compare:
            CompetitorScorecard().render({args.suite: report}, track=args.suite, model_name="NanoLLM Champion")

if __name__ == "__main__":
    main()
