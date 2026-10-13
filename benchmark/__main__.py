import argparse, os, warnings
import transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

from benchmark import Scorecard
from benchmark.competitors import AVAILABLE_COMPETITORS
from data import Dataset
from inference import DecisionEngine
from model import DEFAULT_CHECKPOINT, TorchModel, get_latest_checkpoint

def main() -> None:
    latest_ckpt, _ = get_latest_checkpoint()
    parser = argparse.ArgumentParser(description="Piqo Benchmark Runner")
    parser.add_argument("--checkpoint", type=str, default=str(latest_ckpt or DEFAULT_CHECKPOINT))
    parser.add_argument("--name", type=str, default="Piqo", help="Model name label in leaderboard")
    parser.add_argument("--play", "-p", action="store_true", help="Interactive playground comparing Piqo with competitors")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test items")
    parser.add_argument("--device", type=str, default=None, help="Target compute device (e.g. 'cuda', 'cpu'; default: auto)")
    parser.add_argument("--competitor", type=str, default="all", help=f"Competitor to benchmark live: {list(AVAILABLE_COMPETITORS.keys())}, 'all', or 'none' (default: all)")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if competitor is already recorded in ledger")
    args = parser.parse_args()

    model = TorchModel(checkpoint_path=args.checkpoint, device=args.device)
    engine = DecisionEngine(model=model)

    if args.play:
        from inference import Choice, ChoiceResult, ProfilingLayer
        print("Loading models...")
        engines = [(args.name, engine | ProfilingLayer())]
        targets = list(AVAILABLE_COMPETITORS.keys()) if args.competitor == "all" else [args.competitor]
        warmup_q = [Choice("warmup", ["yes", "no"])]
        for c_name in targets:
            if c_name in AVAILABLE_COMPETITORS:
                try:
                    c_eng = AVAILABLE_COMPETITORS[c_name](device=args.device) | ProfilingLayer()
                    c_eng.decide("warmup", warmup_q)  # warm up so first run isn't skewed by download/init
                    engines.append((c_name.capitalize(), c_eng))
                except Exception as e:
                    print(f"Skipping competitor '{c_name}': {e}")

        names = ", ".join(n for n, _ in engines)
        print(f"\nPiqo Benchmark Playground [{names}]. Enter query (or 'q' to quit):")
        while True:
            try:
                ctx = input("\nContext: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not ctx:
                continue
            if ctx.lower() in ("q", "quit", "exit"):
                break

            try:
                raw_choices = input("Choices (comma-separated): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            opts = [o.strip() for o in raw_choices.split(",") if o.strip()]
            if not opts:
                continue

            qs = [Choice("decision", opts)]
            print(f"\nState: {ctx}\n" + "-" * 55)
            for name, eng in engines:
                try:
                    res = eng.decide(ctx, qs)
                    ans = res.answers.get("decision")
                    if isinstance(ans, ChoiceResult):
                        print(f"  {name:<15s} -> {ans.choice:<16s} ({ans.confidence * 100:5.1f}%) | {res.latency_ms:6.2f} ms")
                    else:
                        print(f"  {name:<15s} -> {str(ans):<16s} | {res.latency_ms:6.2f} ms")
                except Exception as e:
                    print(f"  {name:<15s} -> Error: {e}")
        return

    card = Scorecard()
    dataset = Dataset()
    items = dataset.load(args.split)
    if args.limit:
        items = items[:args.limit]

    print(f"Benchmarking {args.name} on {len(items)} items ({args.split} split)...")
    card.evaluate(engine, items, model_name=args.name, force=args.force)

    if args.competitor and args.competitor != "none":
        targets = list(AVAILABLE_COMPETITORS.keys()) if args.competitor == "all" else [args.competitor]
        for c_name in targets:
            if c_name in AVAILABLE_COMPETITORS:
                cls = AVAILABLE_COMPETITORS[c_name]
                try:
                    card.evaluate(cls(device=args.device), items, model_name=c_name.capitalize(), force=args.force)
                except Exception as e:
                    print(f"Skipping live run for competitor '{c_name}': {e}")

    card.render()

if __name__ == "__main__":
    main()
