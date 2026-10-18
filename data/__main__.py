from collections import Counter
from typing import Optional
from rich import box
from rich.console import Console
from rich.table import Table
import typer

from data.dataset import Dataset
from data.policies.leakage import NGramLeakagePolicy
from data.split.split import BudgetSplitPolicy

app = typer.Typer(add_completion=False, help="Piqo Dataset Corpus Builder CLI")

@app.command()
def main(
    seed: int = typer.Option(42, help="Random seed"),
    train_cap: int = typer.Option(1500, "--train-cap", "-t", help="Max training cases per category"),
    val_cap: int = typer.Option(150, "--val-cap", "-v", help="Max validation cases per category"),
    test_cap: int = typer.Option(400, "--test-cap", help="Max test holdout cases per category"),
    n_gram: int = typer.Option(10, help="N-gram leakage window"),
) -> None:
    console = Console()
    dataset = Dataset(
        split_policy=BudgetSplitPolicy(
            max_train_per_category=train_cap,
            max_val_per_category=val_cap,
            max_test_per_category=test_cap,
            seed=seed,
        ),
        leakage_policy=NGramLeakagePolicy(n_gram=n_gram),
    )
    console.print(f"\n[bold cyan]Piqo Data Pipeline[/bold cyan]\n")
    console.print("[dim]Ingesting sources, partitioning splits & auditing zero-leakage invariant...[/dim]\n")
    splits = dataset.prepare()

    all_cases = list(splits.train) + list(splits.val) + list(splits.test)
    source_counts = Counter((c.source or "unknown", c.category or "general") for c in all_cases)

    src_table = Table(title="Corpus Sources", box=box.HORIZONTALS, title_style="bold magenta")
    src_table.add_column("Source", style="cyan")
    src_table.add_column("Category", style="yellow")
    src_table.add_column("Count", justify="right", style="green")

    for (src_name, cat), count in sorted(source_counts.items()):
        src_table.add_row(src_name, cat, f"{count:,}")

    console.print(src_table)
    console.print(f"Total Prepared Records: [bold green]{len(all_cases):,}[/bold green]\n")

    summary_table = Table(title="Corpus Summary & Leakage Scorecard", box=box.HORIZONTALS, title_style="bold white")
    summary_table.add_column("Metric / Split", style="cyan")
    summary_table.add_column("Cases", justify="right", style="bold green")
    summary_table.add_column("Invariant", justify="right", style="dim")
    summary_table.add_row("Train Split", f"{len(splits.train):,}", "-")
    summary_table.add_row("Validation Split", f"{len(splits.val):,}", "-")
    summary_table.add_row("Test Split", f"{len(splits.test):,}", "-")
    summary_table.add_row("Pruned Train Leaks", f"{dataset.leak_audit.get('pruned_train', 0):,}", "-")
    summary_table.add_row("Pruned Val Leaks", f"{dataset.leak_audit.get('pruned_val', 0):,}", "-")
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
