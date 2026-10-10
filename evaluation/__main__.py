import argparse
from data import Dataset
from evaluation import IEvaluator, Scorecard
from inference import DecisionEngine, ProfilingLayer
from training import DEFAULT_CHECKPOINT, get_latest_checkpoint

def main() -> None:
    latest_ckpt, _ = get_latest_checkpoint()
    parser = argparse.ArgumentParser(description="Piqo Evaluation Benchmark Runner")
    parser.add_argument("--checkpoint", type=str, default=str(latest_ckpt or DEFAULT_CHECKPOINT))
    parser.add_argument("--track", type=str, default="benchmark", help="Track name label for scorecard")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test items")
    args = parser.parse_args()

    engine = DecisionEngine.from_checkpoint(args.checkpoint) | ProfilingLayer()
    card: IEvaluator = Scorecard()

    dataset = Dataset()
    items = dataset.load("test")
    if args.limit:
        items = items[:args.limit]

    print(f"Running evaluation benchmark on {len(items)} holdout items...")
    report = card.evaluate(engine, items, track_name=args.track)
    card.render({args.track: report}, model_name="Piqo")

if __name__ == "__main__":
    main()
