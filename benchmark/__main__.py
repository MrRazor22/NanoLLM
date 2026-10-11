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
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if competitor is already recorded in ledger")
    args = parser.parse_args()

    card = Scorecard()
    dataset = Dataset()
    items = dataset.load(args.split)
    if args.limit:
        items = items[:args.limit]

    model = TorchModel(checkpoint_path=args.checkpoint)
    engine = DecisionEngine(model=model) | ProfilingLayer()

    print(f"Benchmarking {args.name} on {len(items)} items ({args.split} split)...")
    card.evaluate(engine, items, model_name=args.name, force=args.force)

    if args.competitor and args.competitor != "none":
        targets = list(AVAILABLE_COMPETITORS.keys()) if args.competitor == "all" else [args.competitor]
        for c_name in targets:
            if c_name in AVAILABLE_COMPETITORS:
                cls = AVAILABLE_COMPETITORS[c_name]
                try:
                    card.evaluate(cls(), items, model_name=c_name.capitalize(), force=args.force)
                except Exception as e:
                    print(f"Skipping live run for competitor '{c_name}': {e}")

    card.render()

if __name__ == "__main__":
    main()
