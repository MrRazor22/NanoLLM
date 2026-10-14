import os, warnings
from typing import Optional
import transformers
import typer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

from benchmark.scorecard import Scorecard
from benchmark.competitors.gliner_decide import GlinerDecideEngine
from benchmark.competitors.kev import KevEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine
from benchmark.suites.suite import IBenchmarkSuite
from benchmark.suites.corpus_holdout import CorpusHoldoutSuite
from benchmark.suites.laya_typed import LayaTypedDecisionsSuite
from benchmark.suites.kev_transfer import KevTransferSuite
from benchmark.suites.jevbench import JevBenchSuite
from benchmark.suites.fast_decisions import FastDecisionsSuite
from benchmark.suites.verdict_nli import VerdictNliSuite
from data.dataset import Dataset
from inference.decision_engine import DecisionEngine

AVAILABLE_COMPETITORS = {
    "laya": LayaEngine,
    "verdict": VerdictEngine,
    "kev": KevEngine,
    "gliner": GlinerDecideEngine,
}

AVAILABLE_SUITES = {
    "all": CorpusHoldoutSuite,
    "corpus": CorpusHoldoutSuite,
    "laya": LayaTypedDecisionsSuite,
    "kev": KevTransferSuite,
    "jev": JevBenchSuite,
    "gliner": FastDecisionsSuite,
    "verdict": VerdictNliSuite,
}

app = typer.Typer(add_completion=False, help="Piqo Benchmark Runner CLI")

@app.command()
def main(
    checkpoint: Optional[str] = typer.Option(None, help="Model checkpoint path"),
    name: str = typer.Option("Piqo", help="Model label"),
    play: bool = typer.Option(False, "--play", "-p", help="Playground mode"),
    suite: str = typer.Option("all", help="Benchmark suite (all, corpus, laya, kev, jev, gliner, verdict)"),
    split: str = typer.Option("test", help="Dataset split for internal suite"),
    limit: Optional[int] = typer.Option(None, help="Item limit"),
    device: Optional[str] = typer.Option(None, help="Target compute device"),
    competitor: str = typer.Option("all", help="Competitor to benchmark"),
    force: bool = typer.Option(False, "--force", help="Force re-evaluation"),
) -> None:
    engine = DecisionEngine(model=checkpoint, device=device)

    if play:
        from inference.questions import Choice, ChoiceResult
        from inference.profiling_layer import ProfilingLayer
        print("Loading models...")
        engines = [(name, engine | ProfilingLayer())]
        targets = list(AVAILABLE_COMPETITORS.keys()) if competitor == "all" else [competitor]
        warmup_q = [Choice("warmup", ["yes", "no"])]
        for c_name in targets:
            if c_name in AVAILABLE_COMPETITORS:
                try:
                    c_eng = AVAILABLE_COMPETITORS[c_name](device=device) | ProfilingLayer()
                    c_eng.decide("warmup", warmup_q)
                    engines.append((c_name.capitalize(), c_eng))
                except Exception as e:
                    print(f"Skipping competitor '{c_name}': {e}")

        names = ", ".join(n for n, _ in engines)
        print(f"\nPiqo Benchmark Playground [{names}]. Enter query (or 'q' to quit):")
        while True:
            try:
                ctx = input("\nContext: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not ctx:
                continue
            if ctx.lower() in ("q", "quit", "exit"):
                break

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
                    if isinstance(ans, ChoiceResult):
                        print(f"  {m_name:<15s} -> {ans.choice:<16s} ({ans.confidence * 100:5.1f}%) | {res.latency_ms:6.2f} ms")
                    else:
                        print(f"  {m_name:<15s} -> {str(ans):<16s} | {res.latency_ms:6.2f} ms")
                except Exception as e:
                    print(f"  {m_name:<15s} -> Error: {e}")
        return

    card = Scorecard()
    suite_cls = AVAILABLE_SUITES.get(suite, CorpusHoldoutSuite)
    active_suite = suite_cls(split=split) if suite_cls is CorpusHoldoutSuite else suite_cls()
    items = list(active_suite.load_cases())
    if limit:
        items = items[:limit]

    print(f"Benchmarking {name} on {len(items)} items [Suite: {active_suite.name}]...")
    if active_suite.published_baselines:
        print("Published Baselines:")
        for baseline_name, score in active_suite.published_baselines.items():
            print(f"  {baseline_name:<22}: {score * 100:.1f}%")

    card.evaluate(engine, items, model_name=name, force=force)

    if competitor and competitor != "none":
        targets = list(AVAILABLE_COMPETITORS.keys()) if competitor == "all" else [competitor]
        for c_name in targets:
            if c_name in AVAILABLE_COMPETITORS:
                cls = AVAILABLE_COMPETITORS[c_name]
                try:
                    card.evaluate(cls(device=device), items, model_name=c_name.capitalize(), force=force)
                except Exception as e:
                    print(f"Skipping live run for competitor '{c_name}': {e}")

    card.render()

if __name__ == "__main__":
    app()
