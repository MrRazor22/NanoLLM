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
    def __or__(self, layer: Callable[["IDecisionEngine"], "IDecisionEngine"]) -> "IDecisionEngine": ...

class DecisionEngine(IDecisionEngine):
    """Core Primitive Implementation: Coordinates neural prediction and decision resolution."""

    def __init__(self, model: Optional["IModel"] = None, resolver: Optional[IResolver] = None):
        if model is None:
            from model.torch_model import TorchModel
            model = TorchModel()
        self.model, self.resolver = model, resolver or DecisionResolver()

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        pred = self.model.predict(state, questions)
        return DecisionResult(answers=self.resolver.resolve(questions, pred.slots, pred.scores))

    def __or__(self, layer: Callable[["IDecisionEngine"], "IDecisionEngine"]) -> "IDecisionEngine":
        return layer(self)

    @classmethod
    def from_weights(
        cls,
        source: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        resolver: Optional[IResolver] = None,
    ) -> IDecisionEngine:
        from model.model_extensions import load_weights
        from model.torch_model import TorchModel

        m = TorchModel(device=device)
        load_weights(m, source) if source is not None else load_weights(m)
        return cls(model=m, resolver=resolver)


