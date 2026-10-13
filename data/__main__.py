from typing import Optional
import typer

from data.dataset import CORPUS_DIR, Dataset
from data.curation.budget_split import BudgetSplitPolicy
from data.curation.ngram_leakage import NGramLeakagePolicy
from data.disk_cache import DiskCachePolicy

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
    dataset = Dataset(
        split_policy=BudgetSplitPolicy(
            train_quota=train_quota,
            val_quota=val_quota,
            test_quota=test_quota,
            seed=seed,
        ),
        leakage_policy=NGramLeakagePolicy(n_gram=n_gram),
        cache_policy=DiskCachePolicy(root_dir=corpus_dir),
    )

    print(f"Building dataset corpus into {corpus_dir}...")
    splits = dataset.prepare()
    print(f"Corpus ready: {len(splits.train)} train, {len(splits.val)} val, {len(splits.test)} test records (zero leakage verified).")

if __name__ == "__main__":
    app()
