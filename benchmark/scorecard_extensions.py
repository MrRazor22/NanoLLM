from typing import Any, Dict, Iterable, List, Optional, Sequence
from rich import box
from rich.console import Console
from rich.table import Table

def _order_models(models: Iterable[str]) -> List[str]:
    """Sorts model names deterministically, prioritizing Piqo as the lead column."""
    ordered = sorted(models)
    if "Piqo" in ordered:
        ordered = ["Piqo"] + [m for m in ordered if m != "Piqo"]
    return ordered

def render_suite(data: Dict[str, dict], suite_name: str) -> None:
    """Renders a single benchmark suite leaderboard table using Rich."""
    if not data:
        return
    tracks_data = {k: v for k, v in data.items() if isinstance(v, dict) and k not in ("signature", "count")}
    models = _order_models({m for entry in tracks_data.values() for m in entry if m not in ("count", "signature")})
    if not models:
        return

    tracks = [t for t in sorted(tracks_data.keys()) if t not in ("OVERALL", "LATENCY_P50_MS")]
    for m in ("OVERALL", "LATENCY_P50_MS"):
        if m in tracks_data:
            tracks.append(m)

    console = Console()
    table = Table(
        title=f"Benchmark Suite: {suite_name.upper()}",
        box=box.HORIZONTALS,
        header_style="bold magenta",
        title_style="bold white",
    )
    table.add_column("Track / Metric", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right", style="dim")
    for m in models:
        table.add_column(m, justify="right", style="bold green" if m == "Piqo" else "white")

    for track in tracks:
        entry = tracks_data.get(track, {})
        cnt = str(entry.get("count", "-"))
        if track == "LATENCY_P50_MS":
            cols = [f"{entry[m]:.2f} ms" if m in entry else "-" for m in models]
        else:
            cols = [f"{entry[m]*100:.1f}%" if m in entry else "-" for m in models]
        is_summary = track in ("OVERALL", "LATENCY_P50_MS")
        table.add_row(
            f"[bold]{track}[/bold]" if is_summary else track,
            cnt,
            *cols,
            end_section=is_summary,
        )
    console.print("\n", table, "\n")

def render_summary(ledger: Dict[str, dict]) -> None:
    """Renders a high-level cross-suite executive summary table."""
    if not ledger:
        return
    suites = [s for s, d in ledger.items() if isinstance(d, dict)]
    all_models = {
        m for s_data in ledger.values() if isinstance(s_data, dict)
        for entry in s_data.values() if isinstance(entry, dict)
        for m in entry if m not in ("count", "signature")
    }
    if not all_models:
        return

    sorted_models = _order_models(all_models)
    console = Console()
    table = Table(
        title="EXECUTIVE BENCHMARK SUMMARY (OVERALL ACCURACY)",
        box=box.DOUBLE_EDGE,
        header_style="bold cyan",
        title_style="bold yellow",
    )
    table.add_column("Benchmark Suite", style="bold white")
    for m in sorted_models:
        table.add_column(m, justify="right", style="bold green" if m == "Piqo" else "white")

    for s_name in sorted(suites):
        overall = ledger[s_name].get("OVERALL", {})
        row = [f"{score*100:.1f}%" if isinstance(score := overall.get(m), (int, float)) else "-" for m in sorted_models]
        table.add_row(s_name, *row)

    console.print("\n", table, "\n")

