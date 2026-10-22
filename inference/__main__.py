import logging
from pathlib import Path
from typing import Optional
import typer

from inference.decision_engine import DecisionEngine, IDecisionEngine
from inference.playground import Playground
from inference.profiling_layer import ProfilingLayer
from model.torch_model import TorchModel
from model.weights import list_weights

app = typer.Typer(add_completion=False, help="Piqo Interactive Playground")

@app.command()
def main(
    weights: Optional[str] = typer.Option(None, "--weights", "-w", help="Weights checkpoint path/name (comma-separated for comparison, or 'list')"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device override"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    if weights == "list":
        for c in list_weights():
            print(f"{c.name:<30} {c.size_mb:>6.1f} MB  saved: {c.saved_at or '-'}")
        return

    targets = [w.strip() for w in weights.split(",") if w.strip()] if weights else [None]
    engines: dict[str, IDecisionEngine] = {}

    for target in targets:
        model = TorchModel(device=device)
        if target:
            model.load(target)
        label = f"Piqo ({Path(target).stem})" if target and len(targets) > 1 else "Piqo"
        engines[label] = DecisionEngine(model) | ProfilingLayer()

    Playground().start(engines)

if __name__ == "__main__":
    app()
