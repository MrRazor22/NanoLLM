from typing import Optional
import typer

from data.dataset import Dataset
from data.dataset_extensions import CORPUS_DIR
from data.curation.budget_split import BudgetSplitPolicy
from data.curation.ngram_leakage import NGramLeakagePolicy
from data.storage_layer import DiskStorageLayer

app = typer.Typer(add_completion=False, help="Piqo Dataset Corpus Builder CLI")

@app.command()
def main(
    corpus_dir: str = typer.Option(str(CORPUS_DIR), help="Path to corpus directory"),
    seed: int = typer.Option(42, help="Random seed"),
    train_quota: int = typer.Option(40000, help="Train quota"),
    val_quota: int = typer.Option(2500, help="Val quota"),
    test_quota: Optional[int] = typer.Option(None, help="Test quota"),
    n_gram: int = typer.Option(10, help="N-gram leakage window"),
) -> None:
    inner = Dataset(
        split_policy=BudgetSplitPolicy(
            train_quota=train_quota,
            val_quota=val_quota,
            test_quota=test_quota,
            seed=seed,
        ),
        leakage_policy=NGramLeakagePolicy(n_gram=n_gram),
    )
    dataset = DiskStorageLayer(inner=inner, corpus_dir=corpus_dir)

    print(f"\n[Piqo Data Pipeline] Preparing corpus into: {corpus_dir}")
    print(f"Sources Registered: {len(inner.sources)}")
    print("-" * 75)
    print(f"| {'Source Name':<35} | {'Category':<22} | {'Read Count':>10} |")
    print("-" * 75)

    source_counts = []
    for src in inner.sources:
        recs = src.read()
        cat = recs[0].sample.category if recs else "unknown"
        source_counts.append((src.name, cat, len(recs)))
        print(f"| {src.name:<35} | {cat:<22} | {len(recs):>10} |")

    print("-" * 75)
    total_raw = sum(c[2] for c in source_counts)
    print(f"Total Ingested Records: {total_raw}")

    print("\nPartitioning splits & auditing zero-leakage invariant...")
    splits = dataset.prepare()

    print("\n" + "=" * 75)
    print("Corpus Ready")
    print("=" * 75)
    print(f"  * Train Split:      {len(splits.train):>6} cases")
    print(f"  * Validation Split: {len(splits.val):>6} cases")
    print(f"  * Test Split:       {len(splits.test):>6} cases")
    print("-" * 75)
    print(f"Leakage Audit Scorecard:")
    print(f"  * Pruned Overlaps:  {inner.leak_audit.get('total_unique', 0):>6} cases")
    print(f"  * Test Leaks:       {inner.leak_audit.get('test_leaks', 0):>6} (Invariant: 0)")
    print(f"  * Val Leaks:        {inner.leak_audit.get('val_leaks', 0):>6} (Invariant: 0)")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    app()
