import os, warnings
from pathlib import Path
from typing import List, Optional
import typer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from inference.questions import Choice, ChoiceResult, DecisionResult, Noul, NoulResult, Question, Score, ScoreResult
from inference.decision_engine import DecisionEngine
from inference.profiling_layer import ProfilingLayer
from model.layouts.checkpoints import CHECKPOINTS_DIR, DEFAULT_CHECKPOINT, get_latest_checkpoint, list_all_checkpoints
from model.torch_model import TorchModel

app = typer.Typer(add_completion=False, help="Piqo Micro-Decision Engine CLI")

PRESETS = {
    "triage": {
        "default": "Postgres CPU reached 99% and connection pool is exhausted.",
        "questions": [Choice("department", ["Billing", "Infrastructure", "Sales", "Security"]), Noul("urgent", "Is immediate incident response required?"), Score("severity", min_value=0.0, max_value=100.0)],
    },
    "routing": {
        "default": "Find all customer orders placed in the last 24 hours.",
        "questions": [Choice("tool", ["query_database", "search_docs", "send_notification", "none"]), Noul("requires_confirmation"), Score("priority", min_value=1.0, max_value=10.0)],
    },
}

def render_decision(res: DecisionResult, state: str) -> None:
    print(f"\nState: {state}\nLatency: {res.latency_ms:.2f} ms\n" + "-" * 50)
    for k, v in res.answers.items():
        if isinstance(v, ChoiceResult): print(f"  {k:15s} -> {v.choice:<20s} ({v.confidence * 100:5.1f}%)")
        elif isinstance(v, NoulResult): print(f"  {k:15s} -> {'YES' if v.value else 'NO':<20s} (p={v.probability:.3f})")
        elif isinstance(v, ScoreResult): print(f"  {k:15s} -> {v.score:5.1f} / 100.0")
    print()

def _parse_q(arg: str, kind: str) -> Question:
    name, _, spec = arg.partition(":")
    name, spec = name.strip(), spec.strip() or None
    if kind == "choice": return Choice(name, [o.strip() for o in spec.split(",")] if spec else ["Yes", "No"])
    return Noul(name, instruction=spec) if kind == "noul" else Score(name, instruction=spec, min_value=0.0, max_value=100.0)

@app.command()
def main(
    state: Optional[str] = typer.Argument(None),
    play: bool = typer.Option(False, "--play", "-p"),
    models: bool = typer.Option(False, "--models", "-m"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-k"),
    preset: str = typer.Option("triage", "--preset"),
    choice: Optional[List[str]] = typer.Option(None, "--choice"),
    noul: Optional[List[str]] = typer.Option(None, "--noul"),
    score: Optional[List[str]] = typer.Option(None, "--score"),
    device: Optional[str] = typer.Option(None, "--device"),
) -> None:
    if models:
        print(f"\n{'Checkpoint':<32} | {'Size':>8} | {'Val Loss':>9}\n" + "-" * 55)
        for c in list_all_checkpoints():
            print(f"{c.name:<32} | {c.size_mb:>5.1f} MB | {f'{c.val_loss:.4f}' if c.val_loss is not None else '-':>9}")
        return

    latest, _ = get_latest_checkpoint()
    ckpt_path = checkpoint or str(latest or DEFAULT_CHECKPOINT)
    ckpt = Path(ckpt_path)
    if not ckpt.exists() and (CHECKPOINTS_DIR / ckpt.name).exists(): ckpt = CHECKPOINTS_DIR / ckpt.name

    model = TorchModel(checkpoint_path=ckpt, device=device)
    engine = DecisionEngine(model=model) | ProfilingLayer()

    if play:
        print(f"Piqo Playground [{ckpt.name}]. Enter context (or 'q' to quit):")
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
                raw_opts = input("Choices (comma-separated): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            opts = [o.strip() for o in raw_opts.split(",") if o.strip()]
            if not opts:
                continue
            render_decision(engine.decide(ctx, [Choice("decision", opts)]), ctx)
        return

    qs: List[Question] = [_parse_q(c, "choice") for c in (choice or [])] + [_parse_q(n, "noul") for n in (noul or [])] + [_parse_q(s, "score") for s in (score or [])]
    cur_state = state or PRESETS[preset]["default"]
    render_decision(engine.decide(cur_state, qs or PRESETS[preset]["questions"]), cur_state)

if __name__ == "__main__":
    app()
