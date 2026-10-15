import os, warnings
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from typing import Optional
import typer
from benchmark.competitors import AVAILABLE_COMPETITORS
from benchmark.scorecard import Scorecard
from benchmark.scorecard_extensions import render_suite, render_summary, run_playground
from benchmark.suites import PUBLISHED_SUITES
from inference.decision_engine import DecisionEngine, IDecisionEngine
from inference.questions import Choice
from inference.profiling_layer import ProfilingLayer

app = typer.Typer(add_completion=False, help="Piqo Benchmark Runner CLI")

@app.command()
def main(
    checkpoint: Optional[str] = typer.Option(None, help="Model checkpoint path"),
    name: str = typer.Option("Piqo", help="Model label"),
    play: bool = typer.Option(False, "--play", "-p", help="Playground mode"),
    suite: str = typer.Option("all", help="Benchmark suite (all, laya, kev, jev, gliner, verdict)"),
    limit: Optional[int] = typer.Option(None, help="Item limit per suite"),
    device: Optional[str] = typer.Option(None, help="Target compute device"),
    competitor: str = typer.Option("all", help="Competitor to benchmark (all, none, laya, verdict, kev, gliner)"),
    force: bool = typer.Option(False, "--force", help="Force re-evaluation"),
) -> None:
    competitor_keys = list(AVAILABLE_COMPETITORS.keys()) if competitor == "all" else ([competitor] if competitor != "none" else [])
    targets = [(name, name)] + [(c.capitalize(), c) for c in competitor_keys]

    engine_cache: dict[str, IDecisionEngine] = {}
    def resolve_engine(m_key: str) -> IDecisionEngine:
        if m_key not in engine_cache:
            if m_key == name:
                engine_cache[m_key] = DecisionEngine.from_weights(checkpoint, device=device)
            elif m_key.lower() in AVAILABLE_COMPETITORS:
                cls = AVAILABLE_COMPETITORS[m_key.lower()]
                engine_cache[m_key] = cls(device=device)
        return engine_cache[m_key]

    if play:
        print("Loading models for playground...")
        live_engines = []
        for label, key in targets:
            try:
                eng = resolve_engine(key) | ProfilingLayer()
                eng.decide("warmup", [Choice("decision", ["warm"])])
                live_engines.append((label, eng))
            except Exception as e:
                print(f"Skipping '{label}': {e}")
        run_playground(live_engines)
        return

    card = Scorecard()
    selected_suites = list(PUBLISHED_SUITES.values()) if suite == "all" else ([PUBLISHED_SUITES[suite]] if suite in PUBLISHED_SUITES else [PUBLISHED_SUITES["laya"]])

    for suite_cls in selected_suites:
        active_suite = suite_cls()
        items = list(active_suite.load_cases())[:limit] if limit else list(active_suite.load_cases())

        print(f"\n{'=' * 65}\nEvaluating Suite: {active_suite.name.upper()} ({len(items)} items)")
        if active_suite.published_baselines:
            print("Published Baselines:")
            for b_name, b_score in active_suite.published_baselines.items():
                print(f"  {b_name:<24}: {b_score * 100:.1f}%")
        print(f"{'=' * 65}")

        for label, key in targets:
            try:
                card.evaluate(resolve_engine(key), items, model_name=label, suite=active_suite.name, force=force)
            except Exception as e:
                print(f"Skipping run for '{label}': {e}")

        render_suite(card.ledger.all(active_suite.name), active_suite.name)

    if suite == "all":
        render_summary(card.ledger.all())

if __name__ == "__main__":
    app()
