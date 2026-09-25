import argparse
import json
from pathlib import Path
from typing import List
import torch

from nanollm.inference.engine import DEFAULT_CHECKPOINT, DecisionEngine
from benchmark.evaluator import ModelEvaluator
from benchmark.profiling_layer import ProfilingEvaluatorLayer
from benchmark.reporter import print_benchmark_table

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BENCHMARK_DIR / "data" / "laya_benchmark.json"

def main() -> None:
    parser = argparse.ArgumentParser(description="NanoLLM Honest Benchmark")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to checkpoint")
    parser.add_argument("--baseline", type=str, default=None, help="Optional baseline checkpoint to compare")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Path to benchmark data json")
    parser.add_argument("--agentic", action="store_true", help="Run agentic 120-scenario benchmark instead of Laya suite")
    parser.add_argument("--log-interval", type=int, default=200, help="Live step logging interval")
    args = parser.parse_args()

    data_file = (BENCHMARK_DIR / "data" / "benchmark.json") if args.agentic else Path(args.data)
    with open(data_file, "r", encoding="utf-8") as f:
        items = json.load(f)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = DecisionEngine.from_checkpoint(args.checkpoint, device=device)
    evaluator = ModelEvaluator.from_engine("NanoLLM Champion", engine, log_interval=args.log_interval) | ProfilingEvaluatorLayer(warmup_runs=5)
    reports: List[dict] = [evaluator.evaluate(items)]

    if args.baseline:
        base_engine = DecisionEngine.from_checkpoint(args.baseline, device=device)
        base_evaluator = ModelEvaluator.from_engine("NanoLLM Baseline", base_engine, log_interval=args.log_interval) | ProfilingEvaluatorLayer(warmup_runs=5)
        reports.append(base_evaluator.evaluate(items))

    print_benchmark_table(reports)

if __name__ == "__main__":
    main()
