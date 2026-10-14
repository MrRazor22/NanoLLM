from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence, Union, runtime_checkable

from inference.questions import DecisionResult, DecisionState, Question
from inference.decision_resolver import DecisionResolver, IResolver

if TYPE_CHECKING:
    from model.torch_model import IModel

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Sovereign operational boundary for domain decision resolution."""
    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult: ...

class DecisionEngine(IDecisionEngine):
    """Core Primitive Implementation: Coordinates neural prediction and decision resolution."""

    def __init__(
        self,
        model: Optional[Union["IModel", str, Path]] = None,
        resolver: Optional[IResolver] = None,
        device: Optional[str] = None,
    ):
        if model is None:
            from model.torch_model import TorchModel
            m = TorchModel(device=device)
            try:
                m.load()
            except FileNotFoundError:
                pass
            self.model: IModel = m
        elif isinstance(model, (str, Path)):
            from model.torch_model import TorchModel
            m = TorchModel(device=device)
            m.load(model)
            self.model: IModel = m
        else:
            self.model = model
        self.resolver = resolver or DecisionResolver()

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        pred = self.model.predict(state, questions)
        answers = self.resolver.resolve(questions, pred.slots, pred.scores)
        return DecisionResult(answers=answers)

    def __or__(self, layer: Callable[["IDecisionEngine"], "IDecisionEngine"]) -> "IDecisionEngine":
        return layer(self)
