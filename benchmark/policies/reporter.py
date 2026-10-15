from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, runtime_checkable
from rich import box
from rich.console import Console
from rich.table import Table

@runtime_checkable
class IReporterPolicy(Protocol):
    """Sovereign opinion sub-primitive: presents benchmark suite and summary leaderboards."""
    def suite(self, data: Dict[str, dict], suite_name: str) -> None: ...
    def summary(self, ledger: Dict[str, dict], suites: Optional[Sequence[str]] = None) -> None: ...
    def playground(self, engines: Sequence[tuple[str, Any]]) -> None: ...

def order_models(models: Iterable[str]) -> List[str]:
    """Pure domain transform: sorts model identifiers with Piqo pinned first."""
    ordered = sorted(models)
    if "Piqo" in ordered:
        ordered = ["Piqo"] + [m for m in ordered if m != "Piqo"]
    return ordered

class ConsoleReporterPolicy(IReporterPolicy):
    """Rich console presentation policy for formatted benchmark tables and playground."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def suite(self, data: Dict[str, dict], suite_name: str) -> None:
        if not data:
            return
        tracks_data = {k: v for k, v in data.items() if isinstance(v, dict) and k != "count"}
        models = order_models({m for entry in tracks_data.values() for m in entry if m != "count"})
        if not models:
            return

        tracks = [t for t in sorted(tracks_data.keys()) if t not in ("OVERALL", "BRIER_SCORE", "LATENCY_P50_MS")]
        for m in ("OVERALL", "BRIER_SCORE", "LATENCY_P50_MS"):
            if m in tracks_data:
                tracks.append(m)

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
            elif track == "BRIER_SCORE":
                cols = [f"{entry[m]:.4f}" if m in entry else "-" for m in models]
            else:
                cols = [f"{entry[m]*100:.1f}%" if m in entry else "-" for m in models]
            is_summary = track in ("OVERALL", "BRIER_SCORE", "LATENCY_P50_MS")
            table.add_row(f"[bold]{track}[/bold]" if is_summary else track, cnt, *cols, end_section=is_summary)

        self.console.print("\n", table, "\n")

    def summary(self, ledger: Dict[str, dict], suites: Optional[Sequence[str]] = None) -> None:
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

    def playground(self, engines: Sequence[tuple[str, Any]]) -> None:
        from inference.questions import Choice
        names = ", ".join(n for n, _ in engines)
        print(f"\nPiqo Benchmark Playground [{names}]. Enter query (or 'q' to quit):")
        while True:
            try:
                ctx = input("\nContext (or 'q' to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if ctx.lower() in ("q", "quit", "exit"):
                break
            if not ctx:
                continue

            try:
                raw_choices = input("Choices (comma-separated): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            opts = [o.strip() for o in raw_choices.split(",") if o.strip()]
            if not opts:
                continue

            qs = [Choice("decision", opts)]
            print(f"\nState: {ctx}\n" + "-" * 55)
            for m_name, eng in engines:
                try:
                    res = eng.decide(ctx, qs)
                    ans = res.answers.get("decision")
                    val = getattr(ans, "choice", getattr(ans, "value", str(ans)))
                    conf = getattr(ans, "confidence", 1.0)
                    lat = res.latency_ms
                    print(f"[{m_name:<10}] Choice: {val:<20} | Conf: {conf*100:.1f}% | Lat: {lat:.1f}ms")
                except Exception as e:
                    print(f"[{m_name:<10}] Error: {e}")
            print("-" * 55)
