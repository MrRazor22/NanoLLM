import os, warnings
from typing import List, Optional
from rich import box
from rich.console import Console
from rich.table import Table
import typer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from inference.decision_engine import DecisionEngine, IDecisionEngine
from inference.profiling_layer import ProfilingLayer
from inference.questions import Choice, ChoiceResult, DecisionResult, Noul, NoulResult, Question, Score, ScoreResult
from model.model_extensions import list_weights

console = Console()
app = typer.Typer(add_completion=False, help="Piqo Micro-Decision Engine CLI")

def _questions(choice: list[str], noul: list[str], score: list[str]) -> list[Question]:
    return (
        [Choice(c.partition(":")[0].strip(), [o.strip() for o in c.partition(":")[2].split(",") if o.strip()] or ["Yes", "No"]) for c in choice]
        + [Noul(n.partition(":")[0].strip(), instruction=n.partition(":")[2].strip() or None) for n in noul]
        + [Score(s.partition(":")[0].strip(), instruction=s.partition(":")[2].strip() or None) for s in score]
    )

def render(res: DecisionResult, state: str) -> None:
    console.print(f"\n[bold]Context:[/] {state}")
    t = Table(title=f"Micro-Decision (Latency: {res.latency_ms:.2f} ms)", box=box.ROUNDED)
    t.add_column("Question", style="bold cyan")
    t.add_column("Decision", style="bold green")
    t.add_column("Confidence", justify="right")
    t.add_column("Details / Distribution", style="dim")
    for k, v in res.answers.items():
        match v:
            case ChoiceResult(choice=c, confidence=p, probabilities=dist):
                t.add_row(k, c, f"{p*100:.1f}%", ", ".join(f"{o}: {x*100:.1f}%" for o, x in dist.items()))
            case NoulResult(is_positive=pos, probability=p):
                t.add_row(k, "[bold green]YES[/]" if pos else "[bold red]NO[/]", f"{p*100:.1f}%", f"p={p:.4f}")
            case ScoreResult(score=s, normalized=norm):
                t.add_row(k, f"{s:.1f}", f"{norm:.3f}", "range 0-100")
    console.print(t)

@app.command()
def main(
    state: Optional[str] = typer.Argument(None, help="Context/state to evaluate"),
    play: bool = typer.Option(False, "--play", "-p", help="Playground REPL"),
    weights: bool = typer.Option(False, "--weights", "-w", help="List saved weights"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Weights checkpoint path/id"),
    choice: Optional[List[str]] = typer.Option(None, "--choice", "-c", help="Choice slot ('name:opt1,opt2')"),
    noul: Optional[List[str]] = typer.Option(None, "--noul", "-n", help="Noul slot ('name:instruction')"),
    score: Optional[List[str]] = typer.Option(None, "--score", "-s", help="Score slot ('name:instruction')"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device override"),
) -> None:
    if weights:
        t = Table(title="Saved Weights", box=box.ROUNDED)
        for col in ("Identity", "Size", "Val Loss", "Saved At"): t.add_column(col)
        for c in list_weights(): t.add_row(c.name, f"{c.size_mb:.1f} MB", f"{c.val_loss:.4f}" if c.val_loss else "-", c.saved_at or "-")
        console.print(t); return

    engine: IDecisionEngine = DecisionEngine.from_weights(model, device=device) | ProfilingLayer()

    if play or state is None:
        console.print(f"[bold cyan]Piqo Playground[/] [{model or 'piqo'}]. Enter context (or 'q' to quit):")
        while (ctx := console.input("\n[bold]Context[/]: ").strip()) and ctx.lower() not in ("q", "quit", "exit"):
            opts = [o.strip() for o in console.input("[bold]Choices[/]: ").split(",") if o.strip()]
            if opts: render(engine.decide(ctx, [Choice("decision", opts)]), ctx)
        return

    render(engine.decide(state, _questions(choice or [], noul or [], score or []) or [Choice("decision", ["Yes", "No"])]), state)

if __name__ == "__main__":
    app()
