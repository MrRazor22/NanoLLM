import os, warnings
from typing import List, Optional
from rich import box
from rich.console import Console
from rich.table import Table
import typer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from inference.options import ChoiceAnswer, ChoiceOption, IOption, NoulAnswer, NoulOption, ScoreAnswer, ScoreOption
from inference.decision_engine import DecisionEngine, IDecisionEngine
from inference.profiling_layer import ProfilingLayer
from model.weights import list_weights

console = Console()
app = typer.Typer(add_completion=False, help="Piqo Micro-Decision Engine CLI")

def render(opt: IOption, state: str, latency_ms: float = 0.0) -> None:
    console.print(f"\n[bold]Context:[/] {state}")
    t = Table(title=f"Micro-Decision (Latency: {latency_ms:.2f} ms)", box=box.ROUNDED)
    t.add_column("Type", style="bold cyan")
    t.add_column("Decision", style="bold green")
    t.add_column("Confidence", justify="right")
    t.add_column("Details / Distribution", style="dim")
    match opt.answer:
        case ChoiceAnswer(choice=c, confidence=p, probabilities=dist):
            t.add_row("Choice", c, f"{p*100:.1f}%", ", ".join(f"{o}: {x*100:.1f}%" for o, x in dist.items()))
        case NoulAnswer(decision=dec, confidence=p):
            t.add_row("Noul", "[bold green]YES[/]" if dec else "[bold red]NO[/]", f"{p*100:.1f}%", f"p={p:.4f}")
        case ScoreAnswer(score=s, confidence=p):
            t.add_row("Score", f"{s:.1f}", f"{p*100:.1f}%", "")
    console.print(t)

@app.command()
def main(
    state: Optional[str] = typer.Argument(None, help="Context/state to evaluate"),
    play: bool = typer.Option(False, "--play", "-p", help="Playground REPL"),
    weights: bool = typer.Option(False, "--weights", "-w", help="List saved weights"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Weights checkpoint path/id"),
    choice: Optional[List[str]] = typer.Option(None, "--choice", "-c", help="Choice options (comma-separated: 'opt1,opt2')"),
    noul: bool = typer.Option(False, "--noul", "-n", help="Noul binary decision"),
    score: bool = typer.Option(False, "--score", "-s", help="Score continuous decision"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device override"),
) -> None:
    if weights:
        t = Table(title="Saved Weights", box=box.ROUNDED)
        for col in ("Identity", "Size", "Val Loss", "Saved At"): t.add_column(col)
        for c in list_weights(): t.add_row(c.name, f"{c.size_mb:.1f} MB", f"{c.meta.val_loss:.4f}" if c.meta.val_loss else "-", c.meta.saved_at or "-")
        console.print(t); return

    profiler = ProfilingLayer()
    try:
        engine: IDecisionEngine = DecisionEngine(checkpoint=model, device=device) | profiler
    except FileNotFoundError as e:
        console.print(f"[bold red]Model Error:[/] {e}")
        available = [w.name for w in list_weights()]
        if available:
            console.print(f"Available weights in cache: {available}")
        raise typer.Exit(code=1)

    if play or state is None:
        console.print(f"[bold cyan]Piqo Playground[/] [{model or 'piqo'}]. Enter context (or 'q' to quit):")
        while (ctx := console.input("\n[bold]Context[/]: ").strip()) and ctx.lower() not in ("q", "quit", "exit"):
            opts = [o.strip() for o in console.input("[bold]Choices[/]: ").split(",") if o.strip()]
            if opts:
                resolved = engine.decide(ctx, ChoiceOption(options=opts))
                render(resolved, ctx, profiler.last_latency_ms)
        return

    opt: IOption
    if choice:
        opts = [o.strip() for item in choice for o in item.split(",") if o.strip()]
        opt = ChoiceOption(options=opts)
    elif noul:
        opt = NoulOption()
    elif score:
        opt = ScoreOption()
    else:
        opt = ChoiceOption(options=["Yes", "No"])

    resolved = engine.decide(state, opt)
    render(resolved, state, profiler.last_latency_ms)

if __name__ == "__main__":
    app()
