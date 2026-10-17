from collections import Counter
from typing import Optional
from rich import box
from rich.console import Console
from rich.table import Table
import typer

from data.dataset import Dataset
from data.curation.budget_split import BudgetSplitPolicy
from data.curation.ngram_leakage import NGramLeakagePolicy

app = typer.Typer(add_completion=False, help="Piqo Dataset Corpus Builder CLI")

@app.command()
def main(
    seed: int = typer.Option(42, help="Random seed"),
    train_quota: int = typer.Option(40000, help="Train quota"),
    train_quota_per_category: Optional[int] = typer.Option(1500, help="Train quota per category"),
    val_quota: int = typer.Option(2500, help="Val quota"),
    test_quota: Optional[int] = typer.Option(None, help="Test quota"),
    n_gram: int = typer.Option(10, help="N-gram leakage window"),
) -> None:
    console = Console()
    dataset = Dataset(
        split_policy=BudgetSplitPolicy(
            train_quota=train_quota,
            train_quota_per_category=train_quota_per_category,
            val_quota=val_quota,
            test_quota=test_quota,
            seed=seed,
        ),
        leakage_policy=NGramLeakagePolicy(n_gram=n_gram),
    )
    console.print(f"\n[bold cyan]Piqo Data Pipeline[/bold cyan]\n")

    src_table = Table(title="Registered Sources", box=box.HORIZONTALS, title_style="bold magenta")
    src_table.add_column("Source Name", style="cyan")
    src_table.add_column("Category", style="yellow")
    src_table.add_column("Read Count", justify="right", style="green")

    source_counts = []
    total_src = len(dataset.sources)
    for i, src in enumerate(dataset.sources, 1):
        console.print(f"[{i:02d}/{total_src:02d}] Ingesting {src.name}...", end="\r")
        recs = src.read()
        cat = recs[0].sample.category if recs else "unknown"
        source_counts.append((src.name, cat, len(recs)))
        src_table.add_row(src.name, cat, f"{len(recs):,}")

    console.print(src_table)
    console.print(f"Total Ingested Records: [bold green]{sum(c[2] for c in source_counts):,}[/bold green]\n")

    console.print("[dim]Partitioning splits & auditing zero-leakage invariant...[/dim]\n")
    splits = dataset.prepare()

    summary_table = Table(title="Corpus Summary & Leakage Scorecard", box=box.HORIZONTALS, title_style="bold white")
    summary_table.add_column("Metric / Split", style="cyan")
    summary_table.add_column("Cases", justify="right", style="bold green")
    summary_table.add_column("Invariant", justify="right", style="dim")
    summary_table.add_row("Train Split", f"{len(splits.train):,}", "-")
    summary_table.add_row("Validation Split", f"{len(splits.val):,}", "-")
    summary_table.add_row("Test Split", f"{len(splits.test):,}", "-")
    summary_table.add_row("Pruned Overlaps", f"{dataset.leak_audit.get('total_unique', 0):,}", "-")
    summary_table.add_row("Test Leaks", f"{dataset.leak_audit.get('test_leaks', 0)}", "[bold green]0[/bold green]")
    summary_table.add_row("Val Leaks", f"{dataset.leak_audit.get('val_leaks', 0)}", "[bold green]0[/bold green]")
    console.print(summary_table)

    dist_table = Table(title="Train Category Distribution", box=box.HORIZONTALS, title_style="bold white")
    dist_table.add_column("Category", style="yellow")
    dist_table.add_column("Cases", justify="right", style="green")
    dist_table.add_column("Share", justify="right", style="cyan")
    train_counts = Counter(c.category or "general" for c in splits.train)
    total_train = len(splits.train)
    for cat, cnt in train_counts.most_common():
        pct = (cnt / max(1, total_train)) * 100
        dist_table.add_row(cat, f"{cnt:,}", f"{pct:.1f}%")
    console.print(dist_table)
    console.print("")

if __name__ == "__main__":
    app()
