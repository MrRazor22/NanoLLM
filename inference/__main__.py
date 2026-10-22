import logging
from typing import List, Optional
import typer

from inference.options import (
    IOption, ChoiceOption, NoulOption, ScoreOption,
    IAnswer, ChoiceAnswer, NoulAnswer, ScoreAnswer,
)
from inference.decision_engine import IDecisionEngine, DecisionEngine
from inference.profiling_layer import ProfilingLayer
from model.torch_model import TorchModel
from model.weights import list_weights

app = typer.Typer(add_completion=False, help="Piqo Decision Engine CLI")

def format_decision(answer: IAnswer) -> str:
    match answer:
        case ChoiceAnswer(choice=choice, confidence=conf):
            return f"Choice: {choice} ({conf*100:.1f}%)"
        case NoulAnswer(decision=dec, confidence=conf):
            return f"Noul: {'YES' if dec else 'NO'} ({conf*100:.1f}%)"
        case ScoreAnswer(score=score, confidence=conf):
            return f"Score: {score:.1f} ({conf*100:.1f}%)"
        case _:
            return f"{type(answer).__name__} ({answer.confidence*100:.1f}%)"

def parse_option(state: str, choice: Optional[List[str]], noul: bool, score: bool) -> IOption:
    if choice:
        items = [o.strip() for item in choice for o in item.split(",") if o.strip()]
        return ChoiceOption(instruction=state or "Select option", criteria={o: "" for o in items})
    if score:
        return ScoreOption(instruction=state or "Rate score", criteria=["Low", "Medium", "High"])
    return NoulOption(instruction=state or "holds")

@app.command()
def main(
    state: Optional[str] = typer.Argument(None, help="Context/state to evaluate"),
    play: bool = typer.Option(False, "--play", "-p", help="Playground REPL"),
    weights: bool = typer.Option(False, "--weights", "-w", help="List saved weights"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Weights checkpoint path/id"),
    choice: Optional[List[str]] = typer.Option(None, "--choice", "-c", help="Choice options"),
    noul: bool = typer.Option(False, "--noul", "-n", help="Noul binary decision"),
    score: bool = typer.Option(False, "--score", "-s", help="Score continuous decision"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device override"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    if weights:
        for c in list_weights():
            print(f"{c.name:<30} {c.size_mb:>6.1f} MB  loss: {c.meta.get('val_loss', '-'):<8} saved: {c.meta.get('saved_at', '-')}")
        return

    profiler = ProfilingLayer()
    try:
        target_model = TorchModel(device=device)
        if model:
            target_model.load(model)
        engine: IDecisionEngine = DecisionEngine(target_model) | profiler
    except FileNotFoundError as e:
        print(f"Model Error: {e}")
        available = [w.name for w in list_weights()]
        if available:
            print(f"Available weights: {available}")
        raise typer.Exit(code=1)

    if play or state is None:
        print(f"Piqo Playground [{model or 'piqo'}]. Enter context (or 'q' to quit):")
        while (ctx := input("\nContext: ").strip()) and ctx.lower() not in ("q", "quit", "exit"):
            opts = [o.strip() for o in input("Choices: ").split(",") if o.strip()]
            if opts:
                ans = engine.decide(ctx, ChoiceOption(instruction=ctx, criteria={o: "" for o in opts}))
                print(f"Decision: {format_decision(ans)} | Latency: {profiler.last_latency_ms:.2f} ms")
        return

    opt = parse_option(state, choice, noul, score)
    ans = engine.decide(state, opt)
    print(f"Decision: {format_decision(ans)} | Latency: {profiler.last_latency_ms:.2f} ms")

if __name__ == "__main__":
    app()
