import argparse
from harness.dataset.curriculum import build_curriculum

def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt raw sources into training datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--abstention-rate", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--no-sources", action="store_true", help="Skip saving individual source jsonl files")
    args = parser.parse_args()

    print(">>> Starting Training Dataset Adaptation...")
    build_curriculum(
        seed=args.seed,
        abstention_rate=args.abstention_rate,
        val_ratio=args.val_ratio,
        save_individual_sources=not args.no_sources,
    )

if __name__ == "__main__":
    main()
