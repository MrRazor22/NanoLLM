import argparse
import torch

from nanollm.inference.engine import DEFAULT_CHECKPOINT, DecisionEngine
from benchmark.evaluator import ModelEvaluator
from benchmark.profiling_layer import ProfilingEvaluatorLayer
from benchmark.reporter import print_consolidated_scorecard, print_scorecard
from benchmark.suites import SUITES, get_suite

def main() -> None:
    parser = argparse.ArgumentParser(description="NanoLLM Honest Benchmark")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to checkpoint")
    parser.add_argument("--suite", "--track", type=str, default="all", dest="suite", help="Benchmark suite to run (all, agentic, abstention, typed_decisions, laya)")
    parser.add_argument("--all", action="store_const", dest="suite", const="all", help="Run all suites")
    parser.add_argument("--agentic", action="store_const", dest="suite", const="agentic", help="Run agentic suite")
    parser.add_argument("--abstention", action="store_const", dest="suite", const="abstention", help="Run abstention suite")
    parser.add_argument("--laya", action="store_const", dest="suite", const="laya", help="Run laya suite")
    parser.add_argument("--typed-decisions", "--verdict", action="store_const", dest="suite", const="typed_decisions", help="Run typed decisions suite")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for quick validation")
    parser.add_argument("--log-interval", type=int, default=100, help="Live step logging interval")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = DecisionEngine.from_checkpoint(args.checkpoint, device=device)
    evaluator = ModelEvaluator.from_engine("NanoLLM Champion", engine, log_interval=args.log_interval) | ProfilingEvaluatorLayer(warmup_runs=5)

    if args.suite == "all":
        reports = {}
        for name in ["agentic", "abstention", "typed_decisions", "laya"]:
            print(f"\n{'='*70}\n>>> Running Benchmark Suite: {name.upper().replace('_', ' ')}\n{'='*70}", flush=True)
            reports[name] = evaluator.evaluate(get_suite(name), limit=args.limit)
        print_consolidated_scorecard(reports)
    else:
        suite = get_suite(args.suite)
        report = evaluator.evaluate(suite, limit=args.limit)
        print_scorecard(report, track=args.suite)

if __name__ == "__main__":
    main()
