import os, sys, warnings
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from typing import Optional, Sequence
import transformers
import typer
transformers.logging.set_verbosity_error()

from benchmark.scorecard import Scorecard, signature_for
from benchmark.ledger import DiskLedgerPolicy
from benchmark.scorecard_extensions import render_suite, render_summary
from benchmark.competitors.gliner_decide import GlinerDecideEngine
from benchmark.competitors.kev import KevEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine
from benchmark.suites.laya_typed import LayaTypedDecisionsSuite
from benchmark.suites.kev_transfer import KevTransferSuite
from benchmark.suites.jevbench import JevBenchSuite
from benchmark.suites.fast_decisions import FastDecisionsSuite
from benchmark.suites.verdict_nli import VerdictNliSuite
from inference.decision_engine import DecisionEngine, IDecisionEngine
from inference.questions import Choice

AVAILABLE_COMPETITORS = {
    "laya": LayaEngine,
    "verdict": VerdictEngine,
    "kev": KevEngine,
    "gliner": GlinerDecideEngine,
}

PUBLISHED_SUITES = {
    "laya": LayaTypedDecisionsSuite,
    "kev": KevTransferSuite,
    "jev": JevBenchSuite,
    "gliner": FastDecisionsSuite,
    "verdict": VerdictNliSuite,
}

def _run_playground(engines: Sequence[tuple[str, IDecisionEngine]]) -> None:
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
                if ans is None:
                    val = "-"
                elif hasattr(ans, "confidence"):
                    val = f"{str(ans.value):<16s} ({ans.confidence * 100:5.1f}%)"
                else:
                    val = f"{str(ans.value):<16s}"
                print(f"  {m_name:<15s} -> {val} | {res.latency_ms:6.2f} ms")
            except Exception as e:
                print(f"  {m_name:<15s} -> Error: {e}")

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

    engine_cache = {}
    def resolve_engine(m_key: str) -> IDecisionEngine:
        if m_key not in engine_cache:
            if m_key == name:
                engine_cache[m_key] = DecisionEngine(model=checkpoint, device=device)
            elif m_key.lower() in AVAILABLE_COMPETITORS:
                cls = AVAILABLE_COMPETITORS[m_key.lower()]
                engine_cache[m_key] = cls(device=device)
        return engine_cache[m_key]

    if play:
        from inference.profiling_layer import ProfilingLayer
        print("Loading models for playground...")
        live_engines = []
        for label, key in targets:
            try:
                eng = resolve_engine(key) | ProfilingLayer()
                eng.decide("warmup", [Choice("decision", ["warm"])])
                live_engines.append((label, eng))
            except Exception as e:
                print(f"Skipping '{label}': {e}")
        _run_playground(live_engines)
        return

    card = Scorecard()
    ledger = DiskLedgerPolicy()
    selected_suites = list(PUBLISHED_SUITES.values()) if suite == "all" else ([PUBLISHED_SUITES[suite]] if suite in PUBLISHED_SUITES else [LayaTypedDecisionsSuite])

    for suite_cls in selected_suites:
        active_suite = suite_cls()
        items = list(active_suite.load_cases())
        if limit:
            items = items[:limit]
        sig = signature_for(items)

        print(f"\n{'=' * 65}")
        print(f"Evaluating Suite: {active_suite.name.upper()} ({len(items)} items)")
        if active_suite.published_baselines:
            print("Published Baselines:")
            for b_name, b_score in active_suite.published_baselines.items():
                print(f"  {b_name:<24}: {b_score * 100:.1f}%")
        print(f"{'=' * 65}")

        for label, key in targets:
            if force or ledger.get(label, suite=active_suite.name, signature=sig) is None:
                try:
                    eng = resolve_engine(key)
                    report = card.evaluate(eng, items, model_name=label, suite=active_suite.name)
                    ledger.put(report, signature=sig)
                except Exception as e:
                    print(f"Skipping run for '{label}': {e}")

        render_suite(ledger.all(active_suite.name), active_suite.name)

    if suite == "all":
        render_summary(ledger.all())

if __name__ == "__main__":
    app()
