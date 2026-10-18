import os, warnings
from pathlib import Path
from typing import Mapping, Optional, Sequence
import typer

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from benchmark.benchmark import Benchmark
from benchmark.registry import AVAILABLE_ENGINES, BENCHMARK_SUITES
from benchmark.suites.suite import IBenchmarkSuite
from inference.decision_engine import IDecisionEngine
from data.case import ChoiceOption

app = typer.Typer(add_completion=False, help="Piqo Sovereign Benchmark CLI")

def resolve_suites(suite_arg: str) -> Sequence[IBenchmarkSuite]:
    """Resolves CLI suite argument into sovereign IBenchmarkSuite primitives."""
    arg = suite_arg.strip().lower()
    if arg in ("all", "all_suites", "official"):
        return [cls() for cls in BENCHMARK_SUITES.values()]
    queries = [s.strip().lower() for s in suite_arg.split(",") if s.strip()]
    matched = [cls for key, cls in BENCHMARK_SUITES.items() if any(q in key for q in queries)]
    if not matched:
        raise typer.BadParameter(f"No benchmark suites matched '{suite_arg}'. Available: {list(BENCHMARK_SUITES.keys())}")
    return [cls() for cls in dict.fromkeys(matched)]

def resolve_engines(
    model_arg: str,
    device: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> Mapping[str, IDecisionEngine]:
    """Resolves CLI model argument into instantiated IDecisionEngine primitives."""
    arg = model_arg.strip().lower()
    if arg == "all":
        keys = list(AVAILABLE_ENGINES.keys())
    else:
        requested = [k.strip() for k in arg.split(",") if k.strip()]
        invalid = [k for k in requested if k not in AVAILABLE_ENGINES]
        if invalid:
            raise typer.BadParameter(f"Unrecognized engine(s): {invalid}. Available: {list(AVAILABLE_ENGINES.keys())}")
        keys = requested
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
    if not engines:
        raise typer.BadParameter(f"No engines could be initialized for '{model_arg}'.")
    return engines

def run_playground(engines: Sequence[tuple[str, IDecisionEngine]]) -> None:
    names = ", ".join(n for n, _ in engines)
    print(f"\nPiqo Benchmark Playground [{names}]. Enter query (or 'q' to quit):")
    while True:
        try:
            ctx = input("\nContext (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if ctx.lower() in ("q", "quit", "exit") or not ctx:
            if not ctx: continue
            break
        try:
            raw_choices = input("Choices (comma-separated): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        opts = [o.strip() for o in raw_choices.split(",") if o.strip()]
        if not opts:
            continue
        opt = ChoiceOption(options=opts)
        print(f"\nState: {ctx}\n" + "-" * 55)
        for m_name, eng in engines:
            try:
                res_opt = eng.decide(ctx, opt)
                ans = res_opt.answer
                val = getattr(ans, "choice", "-")
                conf = getattr(ans, "confidence", 1.0)
                lat = getattr(eng, "last_latency_ms", 0.0)
                print(f"[{m_name:<10}] Choice: {val:<20} | Conf: {conf*100:.1f}% | Lat: {lat:.1f}ms")
            except Exception as e:
                print(f"[{m_name:<10}] Error: {e}")
        print("-" * 55)

@app.command()
def main(
    suite: str = typer.Option("all", "--suite", "-s", help="Benchmark suite(s): all, laya, jev, kev (comma-separated)"),
    model: str = typer.Option("all", "--model", "-m", help="Model(s) to evaluate: all, piqo, gliner, jev, kev, laya, verdict (comma-separated)"),
    report: bool = typer.Option(False, "--report", "-r", help="Show leaderboard report without running models"),
    limit: Optional[int] = typer.Option(None, help="Item limit per suite"),
    device: Optional[str] = typer.Option(None, help="Target compute device (cpu, cuda, mps)"),
    checkpoint: Optional[str] = typer.Option(None, help="Model checkpoint path for Piqo"),
    skip_cached: bool = typer.Option(False, "--skip-cached", help="Skip models that already have scores in ledger"),
    play: bool = typer.Option(False, "--play", "-p", help="Playground mode"),
) -> None:
    bench = Benchmark()
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
        run_playground(live_engines)
        return

    bench.run(active_suites, active_engines, limit=limit, skip_cached=skip_cached)

if __name__ == "__main__":
    app()
