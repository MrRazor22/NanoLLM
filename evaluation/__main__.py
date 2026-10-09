import argparse
import torch
from evaluation import TRACKS, IEvaluator, Scorecard
from inference import DecisionEngine, ProfilingLayer
from training import DEFAULT_CHECKPOINT, get_latest_checkpoint

def main() -> None:
    latest_ckpt, _ = get_latest_checkpoint()
    parser = argparse.ArgumentParser(description="Piqo Evaluation Benchmark Runner")
    parser.add_argument("--checkpoint", type=str, default=str(latest_ckpt or DEFAULT_CHECKPOINT))
    parser.add_argument("--suite", "--track", type=str, default="all", choices=["all"] + list(TRACKS.keys()))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    engine = DecisionEngine.from_checkpoint(args.checkpoint, device=dev) | ProfilingLayer()
    card: IEvaluator = Scorecard()

    targets = TRACKS if args.suite == "all" else {args.suite: TRACKS[args.suite]}
    reports = {}
    for name, src_cls in targets.items():
        src = src_cls()
        print(f"Running track: {src.display_name}...")
        items = src.load(limit=args.limit)
        reports[name] = card.evaluate(engine, items, track_name=name)

    card.render(reports, model_name="Piqo")

if __name__ == "__main__":
    main()
