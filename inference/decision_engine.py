from typing import Callable, Optional, Protocol, Sequence, Union, runtime_checkable

from inference.questions import DecisionResult, DecisionState, Question
from inference.policies.decision_resolver import DecisionResolver, IResolver
from inference.policies.slot_encoder import ILayoutEncoder, SlotEncoder
from model import IModel, TorchModel

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Sovereign operational boundary for domain decision resolution."""
    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult: ...

class DecisionEngine(IDecisionEngine):
    """Core Primitive Implementation: Coordinates layout encoding, neural prediction, and decision resolution."""

    def __init__(
        self,
        model: Optional[IModel] = None,
        encoder: Optional[ILayoutEncoder] = None,
        resolver: Optional[IResolver] = None,
    ):
        self.model = model or TorchModel()
        self.encoder = encoder or SlotEncoder()
        self.resolver = resolver or DecisionResolver()

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        layout = self.encoder.encode(state, questions)
        scores = self.model.predict(layout).scores
        answers = self.resolver.resolve(questions, layout.slots, scores)
        return DecisionResult(answers=answers)

    def __or__(self, layer: Callable[["IDecisionEngine"], "IDecisionEngine"]) -> "IDecisionEngine":
        return layer(self)

__all__ = ["IDecisionEngine", "DecisionEngine"]
