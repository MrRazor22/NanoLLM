import argparse
from pathlib import Path
from data import (
    CORPUS_DIR,
    Dataset,
    BudgetSplitPolicy,
    NGramLeakagePolicy,
    DecisionsSource,
    GenericSource,
    GlaiveSource,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Piqo Dataset Corpus Builder")
    parser.add_argument("--corpus-dir", type=str, default=str(CORPUS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-quota", type=int, default=40000)
    parser.add_argument("--val-quota", type=int, default=2500)
    parser.add_argument("--test-quota", type=int, default=2500)
    parser.add_argument("--n-gram", type=int, default=13)
    args = parser.parse_args()

    split_policy = BudgetSplitPolicy(
        train_quota=args.train_quota,
        val_quota=args.val_quota,
        test_quota=args.test_quota,
        seed=args.seed,
    )
    leakage_policy = NGramLeakagePolicy(n_gram=args.n_gram)

    dataset = Dataset(
        split_policy=split_policy,
        leakage_policy=leakage_policy,
        corpus_dir=Path(args.corpus_dir),
        seed=args.seed,
    )

    # Register default sources
    dataset.register(GlaiveSource(limit=5000, seed=args.seed))
    dataset.register(DecisionsSource(repeat=5))
    dataset.register(GenericSource("mteb/banking77", "intent", "What is the primary customer request?", limit=5000, split="train", tag="unsplit"))
    dataset.register(GenericSource("mteb/banking77", "intent", "What is the primary customer request?", limit=500, split="test", tag="test"))

    print(f"Building dataset corpus into {args.corpus_dir}...")
    splits = dataset.prepare()
    print(f"Corpus ready: {len(splits.train)} train, {len(splits.val)} val, {len(splits.test)} test records (zero leakage verified).")

if __name__ == "__main__":
    main()
