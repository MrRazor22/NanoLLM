from typing import Iterable, List, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable
from rich import box
from rich.console import Console
from rich.table import Table

SuiteData = Mapping[str, Mapping[str, Union[float, int, str]]]
LedgerData = Mapping[str, SuiteData]

@runtime_checkable
class IReporterPolicy(Protocol):
    def render_suite(self, data: SuiteData, suite_name: str) -> None: ...
    def render_summary(self, ledger: LedgerData, suites: Optional[Sequence[str]] = None) -> None: ...

def order_models(models: Iterable[str]) -> List[str]:
    return sorted(models, key=lambda m: (m != "Piqo", m))

class ConsoleReporterPolicy(IReporterPolicy):
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def render_suite(self, data: SuiteData, suite_name: str) -> None:
        if not data:
            return
        tracks_data = {k: v for k, v in data.items() if isinstance(v, dict) and k != "count"}
        models = order_models({m for entry in tracks_data.values() for m in entry if m != "count"})
        if not models:
            return

        summary_keys = ("OVERALL", "BRIER_SCORE", "REGRESSION_MSE", "LATENCY_P50_MS")
        tracks = [t for t in sorted(tracks_data) if t not in summary_keys] + [t for t in summary_keys if t in tracks_data]

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
            elif track in ("BRIER_SCORE", "REGRESSION_MSE"):
                cols = [f"{entry[m]:.4f}" if m in entry else "-" for m in models]
            else:
                cols = [f"{entry[m]*100:.1f}%" if m in entry else "-" for m in models]
            is_summary = track in summary_keys
            table.add_row(f"[bold]{track}[/bold]" if is_summary else track, cnt, *cols, end_section=is_summary)

        self.console.print("\n", table, "\n")

    def render_summary(self, ledger: LedgerData, suites: Optional[Sequence[str]] = None) -> None:
        if not ledger:
            return
        active_suites = [s for s in suites if s in ledger] if suites else [s for s, d in ledger.items() if isinstance(d, dict)]
        if not active_suites:
            return
        all_models = {
            m for s_name in active_suites if isinstance(s_data := ledger.get(s_name), dict)
            for entry in s_data.values() if isinstance(entry, dict)
            for m in entry if m != "count"
        }
        if not all_models:
            return

        sorted_models = order_models(all_models)
        table = Table(
            title="EXECUTIVE BENCHMARK SUMMARY (OVERALL ACCURACY)",
            box=box.DOUBLE_EDGE,
            header_style="bold cyan",
            title_style="bold yellow",
        )
        table.add_column("Benchmark Suite", style="bold white")
        for m in sorted_models:
            table.add_column(m, justify="right", style="bold green" if m == "Piqo" else "white")

        for s_name in sorted(active_suites):
            overall = ledger[s_name].get("OVERALL", {})
            row = [f"{score*100:.1f}%" if isinstance(score := overall.get(m), (int, float)) else "-" for m in sorted_models]
            table.add_row(s_name, *row)

        self.console.print("\n", table, "\n")
