from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable

from data.case import DecisionCase
from inference.options import IOption, option_to_candidates, resolve_decision
from model.torch_model import IModel
import torch

@runtime_checkable
class IDecisionEngineLayer(Protocol):
    """Layer Decorator Contract (lambda: P -> P): Instruments or wraps decision engine execution."""
    def __call__(self, inner: "IDecisionEngine") -> "IDecisionEngine": ...

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Sovereign operational boundary for domain decision resolution."""
    def decide(self, state: str, option: IOption) -> IOption: ...
    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine": ...

class DecisionEngine(IDecisionEngine):
    """Core Primitive Implementation: Coordinates neural prediction and decision resolution."""

    def __init__(
        self,
        model: Optional[IModel] = None,
        device: Optional[str] = None,
        checkpoint: Optional[Union[str, Path]] = None,
    ):
        if model is None:
            from model.torch_model import TorchModel
            target_model = TorchModel(device=device)
            if checkpoint:
                target_model.load(checkpoint)
            model = target_model
        self.model = model

    def decide(self, state: str, option: IOption) -> IOption:
        candidates = option_to_candidates(option)
        with torch.no_grad():
            scores = self.model.forward([DecisionCase(state=state, candidates=candidates)])[0]
        return resolve_decision(option, scores.tolist())

    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine":
        return layer(self)
