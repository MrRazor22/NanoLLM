import argparse
import transformers
from benchmark import Scorecard
from benchmark.competitors import AVAILABLE_COMPETITORS
from data import Dataset
from inference import DecisionEngine, ProfilingLayer
from model import TorchModel
from training import DEFAULT_CHECKPOINT, get_latest_checkpoint

transformers.logging.set_verbosity_error()

def main() -> None:
    latest_ckpt, _ = get_latest_checkpoint()
    parser = argparse.ArgumentParser(description="Piqo Benchmark Runner")
    parser.add_argument("--checkpoint", type=str, default=str(latest_ckpt or DEFAULT_CHECKPOINT))
    parser.add_argument("--name", type=str, default="Piqo", help="Model name label in leaderboard")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test items")
    parser.add_argument("--competitor", type=str, default="all", help=f"Competitor to benchmark live: {list(AVAILABLE_COMPETITORS.keys())}, 'all', or 'none' (default: all)")
    args = parser.parse_args()

    card = Scorecard()
    dataset = Dataset()
    items = dataset.load(args.split)
    if args.limit:
        items = items[:args.limit]

    model = TorchModel(checkpoint_path=args.checkpoint)
    engine = DecisionEngine(model=model) | ProfilingLayer()

    print(f"Benchmarking {args.name} on {len(items)} items ({args.split} split)...")
    report = card.evaluate(engine, items, model_name=args.name)
    card.record(report)

    if args.competitor and args.competitor != "none":
        targets = list(AVAILABLE_COMPETITORS.keys()) if args.competitor == "all" else [args.competitor]
        for c_name in targets:
            if c_name in AVAILABLE_COMPETITORS:
                cls = AVAILABLE_COMPETITORS[c_name]
                print(f"\nBenchmarking competitor '{c_name}' live on {len(items)} items...")
                try:
                    comp_engine = cls()
                    comp_rep = card.evaluate(comp_engine, items, model_name=f"{c_name.capitalize()}")
                    card.record(comp_rep)
                except Exception as e:
                    print(f"Skipping live run for competitor '{c_name}': {e}")

    card.render()

if __name__ == "__main__":
    main()
