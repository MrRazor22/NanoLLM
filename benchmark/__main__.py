import os, warnings
from pathlib import Path
from typing import Mapping, Optional, Sequence
import typer

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from benchmark.benchmark import Benchmark
from benchmark.policies.ledger import DiskLedgerPolicy
from benchmark.registry import AVAILABLE_ENGINES, PUBLISHED_SUITES
from benchmark.suites.suite import IBenchmarkSuite
from inference.decision_engine import IDecisionEngine
from inference.profiling_layer import ProfilingLayer
from inference.questions import Choice

app = typer.Typer(add_completion=False, help="Piqo Sovereign Benchmark CLI")

def resolve_suites(suite_arg: str) -> Sequence[IBenchmarkSuite]:
    """Resolves CLI suite argument into sovereign IBenchmarkSuite primitives."""
    if suite_arg.strip().lower() == "all":
        unique_classes = list(dict.fromkeys(PUBLISHED_SUITES.values()))
        return [cls() for cls in unique_classes]
    keys = [s.strip().lower() for s in suite_arg.split(",") if s.strip().lower() in PUBLISHED_SUITES]
    unique_classes = list(dict.fromkeys([PUBLISHED_SUITES[k] for k in keys]))
    return [cls() for cls in unique_classes] if unique_classes else [PUBLISHED_SUITES["laya"]()]

def resolve_engines(
    model_arg: str,
    device: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> Mapping[str, IDecisionEngine]:
    """Resolves CLI model argument into instantiated IDecisionEngine primitives."""
    arg = model_arg.strip().lower()
    keys = list(AVAILABLE_ENGINES.keys()) if arg == "all" else [k.strip() for k in arg.split(",") if k.strip() in AVAILABLE_ENGINES]
    engines: dict[str, IDecisionEngine] = {}
    for k in keys:
        cls = AVAILABLE_ENGINES[k]
        kwargs = {"device": device}
        if k == "piqo" and checkpoint is not None:
            kwargs["checkpoint"] = checkpoint
        try:
            engines[k.title()] = cls(**kwargs)
        except Exception as e:
            print(f"[{k.title()}] Skipping: could not initialize ({e})")
    return engines

@app.command()
def main(
    suite: str = typer.Option("all", "--suite", "-s", help="Benchmark suite(s): all, gliner, jev, kev, laya (comma-separated)"),
    model: str = typer.Option("all", "--model", "-m", help="Model(s) to evaluate: all, piqo, gliner, jev, kev, laya, verdict (comma-separated)"),
    report: bool = typer.Option(False, "--report", "-r", help="Show leaderboard report without running models"),
    out: Path = typer.Option(Path("benchmark/scores.json"), "--out", "-o", help="Scorecard ledger JSON file path"),
    limit: Optional[int] = typer.Option(None, help="Item limit per suite"),
    device: Optional[str] = typer.Option(None, help="Target compute device (cpu, cuda, mps)"),
    checkpoint: Optional[str] = typer.Option(None, help="Model checkpoint path for Piqo"),
    skip_cached: bool = typer.Option(False, "--skip-cached", help="Skip models that already have scores in ledger"),
    play: bool = typer.Option(False, "--play", "-p", help="Playground mode"),
) -> None:
    bench = Benchmark(ledger=DiskLedgerPolicy(scores_path=out))
    active_suites = resolve_suites(suite)

    if report:
        bench.report(suites=active_suites)
        return

    active_engines = resolve_engines(model, device=device, checkpoint=checkpoint)

    if play:
        live_engines = []
        for name, eng in active_engines.items():
            try:
                profiled = eng | ProfilingLayer()
                profiled.decide("warmup", [Choice("decision", ["warm"])])
                live_engines.append((name, profiled))
            except Exception as e:
                print(f"[{name}] Skipping playground: {e}")
        if not live_engines:
            print("No active engines available for playground.")
            return
        bench.playground(live_engines)
        return

    bench.run(active_suites, active_engines, limit=limit, skip_cached=skip_cached)

if __name__ == "__main__":
    app()
