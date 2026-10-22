import logging
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
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Weights checkpoint path/id"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device override"),
    weights: bool = typer.Option(False, "--weights", "-w", help="List saved weights"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    if weights:
        for c in list_weights():
            print(f"{c.name:<30} {c.size_mb:>6.1f} MB  loss: {c.meta.get('val_loss', '-'):<8} saved: {c.meta.get('saved_at', '-')}")
        return

    target_model = TorchModel(device=device)
    if model:
        target_model.load(model)
    engine: IDecisionEngine = DecisionEngine(target_model) | ProfilingLayer()

    Playground().start({"Piqo": engine})

if __name__ == "__main__":
    app()
