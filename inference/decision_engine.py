from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence, runtime_checkable

from inference.questions import DecisionResult, DecisionState, Question
from inference.decision_resolver import DecisionResolver, IResolver

if TYPE_CHECKING:
    from model import IModel

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Sovereign operational boundary for domain decision resolution."""
    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult: ...

class DecisionEngine(IDecisionEngine):
    """Core Primitive Implementation: Coordinates neural prediction and decision resolution."""

    def __init__(
        self,
        model: Optional["IModel"] = None,
        resolver: Optional[IResolver] = None,
    ):
        if model is None:
            from model.torch_model import TorchModel
            model = TorchModel()
        self.model = model
        self.resolver = resolver or DecisionResolver()

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        pred = self.model.predict(state, questions)
        answers = self.resolver.resolve(questions, pred.slots, pred.scores)
        return DecisionResult(answers=answers)

    def __or__(self, layer: Callable[["IDecisionEngine"], "IDecisionEngine"]) -> "IDecisionEngine":
        return layer(self)

__all__ = ["IDecisionEngine", "DecisionEngine"]
