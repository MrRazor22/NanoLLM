import logging
from collections import Counter
from rich import box
from rich.console import Console
from rich.table import Table
import typer

from data.dataset import Dataset

app = typer.Typer(add_completion=False, help="Piqo Dataset Corpus Builder CLI")

@app.command()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    console = Console()
    console.print("\n[bold cyan]Piqo Data Pipeline[/bold cyan] | Building corpus...\n")
    dataset = Dataset()
    splits = dataset.build()

    total = len(splits.train) + len(splits.val) + len(splits.test)
    pruned = dataset.leak_audit.get("pruned_train", 0) + dataset.leak_audit.get("pruned_val", 0)

    table = Table(title="Corpus Scorecard", box=box.HORIZONTALS, title_style="bold white")
    table.add_column("Split / Metric", style="cyan")
    table.add_column("Cases", justify="right", style="bold green")
    table.add_column("Share", justify="right", style="dim")

    table.add_row("Train", f"{len(splits.train):,}", f"{len(splits.train)/max(1, total)*100:.1f}%")
    table.add_row("Validation", f"{len(splits.val):,}", f"{len(splits.val)/max(1, total)*100:.1f}%")
    table.add_row("Test (Holdout)", f"{len(splits.test):,}", f"{len(splits.test)/max(1, total)*100:.1f}%")
    table.add_row("Pruned Leaks", f"{pruned:,}", "Invariant Clean")
    table.add_row("Total Records", f"{total:,}", "100.0%")
    console.print(table)
    console.print("")

if __name__ == "__main__":
    app()
