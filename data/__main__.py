import argparse
from data import CORPUS_DIR, Dataset

def main() -> None:
    parser = argparse.ArgumentParser(description="Piqo Dataset Corpus Builder")
    parser.add_argument("--corpus-dir", type=str, default=str(CORPUS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-quota", type=int, default=40000)
    parser.add_argument("--val-quota", type=int, default=2500)
    parser.add_argument("--test-quota", type=int, default=2500)
    parser.add_argument("--n-gram", type=int, default=13)
    args = parser.parse_args()

    dataset = Dataset(
        corpus_dir=args.corpus_dir,
        train_quota=args.train_quota,
        val_quota=args.val_quota,
        test_quota=args.test_quota,
        n_gram=args.n_gram,
        seed=args.seed,
    )

    print(f"Building dataset corpus into {args.corpus_dir}...")
    splits = dataset.prepare()
    print(f"Corpus ready: {len(splits.train)} train, {len(splits.val)} val, {len(splits.test)} test records (zero leakage verified).")

if __name__ == "__main__":
    main()
