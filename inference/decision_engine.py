from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable

from inference.questions import DecisionResult, DecisionState, Question
from inference.decision_resolver import DecisionResolver, IResolver
from model.torch_model import IModel

@runtime_checkable
class IDecisionEngineLayer(Protocol):
    """Layer Decorator Contract (lambda: P -> P): Instruments or wraps decision engine execution."""
    def __call__(self, inner: "IDecisionEngine") -> "IDecisionEngine": ...

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Sovereign operational boundary for domain decision resolution."""
    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult: ...
    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine": ...

class DecisionEngine(IDecisionEngine):
    """Core Primitive Implementation: Coordinates neural prediction and decision resolution."""

    def __init__(
        self,
        model: Optional["IModel"] = None,
        device: Optional[str] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        resolver: Optional[IResolver] = None,
    ):
        if model is None:
            from model.torch_model import TorchModel
            from model.weights import has_weights
            model = TorchModel(device=device)
            target_ckpt = checkpoint or ("piqo" if has_weights("piqo") else None)
            if target_ckpt is not None:
                model.load(target_ckpt)
        self.model, self.resolver = model, resolver or DecisionResolver()

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        pred = self.model.predict(state, questions)
        return DecisionResult(answers=self.resolver.resolve(questions, pred.slots, pred.scores))

    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine":
        return layer(self)



