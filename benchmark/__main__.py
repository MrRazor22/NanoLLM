import argparse
from pathlib import Path
import torch

from nanollm.inference.engine import DEFAULT_CHECKPOINT, DecisionEngine
from benchmark.evaluator import ModelEvaluator, load_benchmark_items
from benchmark.profiling_layer import ProfilingEvaluatorLayer
from benchmark.reporter import print_consolidated_scorecard, print_scorecard

TRACKS = ["agentic", "abstention", "typed_decisions", "laya"]

def main() -> None:
    parser = argparse.ArgumentParser(description="NanoLLM Honest Benchmark")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to checkpoint")
    parser.add_argument("--track", type=str, default="all", choices=["all", "agentic", "abstention", "typed_decisions", "laya"], help="Benchmark track to run")
    parser.add_argument("--all", action="store_const", dest="track", const="all", help="Run all benchmark tracks and print consolidated master scorecard")
    parser.add_argument("--typed-decisions", "--verdict", action="store_const", dest="track", const="typed_decisions", help="Alias for --track typed_decisions")
    parser.add_argument("--laya", action="store_const", dest="track", const="laya", help="Alias for --track laya")
    parser.add_argument("--agentic", action="store_const", dest="track", const="agentic", help="Alias for --track agentic")
    parser.add_argument("--abstention", action="store_const", dest="track", const="abstention", help="Alias for --track abstention")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for quick validation")
    parser.add_argument("--log-interval", type=int, default=100, help="Live step logging interval")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = DecisionEngine.from_checkpoint(args.checkpoint, device=device)
    evaluator = ModelEvaluator.from_engine("NanoLLM Champion", engine, log_interval=args.log_interval) | ProfilingEvaluatorLayer(warmup_runs=5)

    if args.track == "all":
        reports = {}
        for trk in TRACKS:
            print(f"\n{'='*70}\n>>> Running Benchmark Track: {trk.upper().replace('_', ' ')}\n{'='*70}", flush=True)
            items = load_benchmark_items(trk, limit=args.limit)
            reports[trk] = evaluator.evaluate(items)
        print_consolidated_scorecard(reports)
    else:
        items = load_benchmark_items(args.track, limit=args.limit)
        report = evaluator.evaluate(items)
        print_scorecard(report, track=args.track)

if __name__ == "__main__":
    main()
