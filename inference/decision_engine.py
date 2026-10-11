from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from inference.questions import DecisionResult, DecisionState, Question
from inference.policies.decision_resolver import DecisionResolver, IResolver
from model import IModel, TorchModel

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Sovereign operational boundary for domain decision resolution."""
    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult: ...

class DecisionEngine(IDecisionEngine):
    """Core Primitive Implementation: Coordinates neural prediction and decision resolution."""

    def __init__(
        self,
        model: Optional[IModel] = None,
        resolver: Optional[IResolver] = None,
    ):
        self.model = model or TorchModel()
        self.resolver = resolver or DecisionResolver()

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        pred = self.model.predict(state, questions)
        answers = self.resolver.resolve(questions, pred.slots, pred.scores)
        return DecisionResult(answers=answers)

    def __or__(self, layer: Callable[["IDecisionEngine"], "IDecisionEngine"]) -> "IDecisionEngine":
        return layer(self)

__all__ = ["IDecisionEngine", "DecisionEngine"]
