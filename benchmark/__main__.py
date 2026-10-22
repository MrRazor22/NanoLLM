import os, warnings
from typing import Optional, Sequence
import typer

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from benchmark.benchmark import Benchmark
from benchmark.playground import Playground
from benchmark.registry import load_engines, load_suites
from benchmark.result import BenchmarkResult

app = typer.Typer(add_completion=False, help="Piqo Benchmark CLI")

def print_results(results: Sequence[BenchmarkResult]) -> None:
    if not results:
        print("No benchmark results to display.")
        return

    print(f"\n{'Suite':<22} | {'Engine':<12} | {'Accuracy':>8} | {'Latency':>10} | {'Brier':>7} | {'Count':>6}")
    print("-" * 77)
    for r in sorted(results, key=lambda x: (x.suite, -x.metrics.accuracy)):
        print(
            f"{r.suite:<22} | {r.engine:<12} | {r.metrics.accuracy * 100:>7.1f}% | "
            f"{r.metrics.latency_p50_ms:>7.1f} ms | {r.metrics.brier_score:>7.4f} | {r.metrics.count:>6}"
        )
    print()

@app.command()
def main(
    suite: str = typer.Option("all", "--suite", "-s", help="Benchmark suite(s) to evaluate"),
    model: str = typer.Option("all", "--model", "-m", help="Engine(s) to evaluate"),
    limit: Optional[int] = typer.Option(None, help="Item limit per suite"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-evaluation, ignoring cache"),
    play: bool = typer.Option(False, "--play", "-p", help="Playground mode"),
) -> None:
    active_engines = load_engines(model)
    if play:
        Playground().start(active_engines)
        return

    active_suites = load_suites(suite)
    results = Benchmark(limit=limit, force=force).run(active_suites, active_engines)
    print_results(results)

if __name__ == "__main__":
    app()
