from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Sequence, Union, runtime_checkable

from inference.questions import DecisionResult, DecisionState, Question
from inference.decision_resolver import DecisionResolver, IResolver

if TYPE_CHECKING:
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
            from model.model_extensions import load_weights
            from model.policies.weight_cache import DiskWeightCache
            from model.torch_model import TorchModel
            model = TorchModel(device=device)
            if checkpoint is not None:
                load_weights(model, checkpoint)
            elif DiskWeightCache().get("piqo") is not None:
                load_weights(model, "piqo")
        self.model, self.resolver = model, resolver or DecisionResolver()

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        pred = self.model.predict(state, questions)
        return DecisionResult(answers=self.resolver.resolve(questions, pred.slots, pred.scores))

    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine":
        return layer(self)

    @classmethod
    def from_weights(
        cls,
        source: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        resolver: Optional[IResolver] = None,
    ) -> IDecisionEngine:
        from model.model_extensions import load_weights
        from model.policies.weight_cache import DiskWeightCache
        from model.torch_model import TorchModel

        m = TorchModel(device=device)
        if source is not None:
            load_weights(m, source)
        elif DiskWeightCache().get("piqo") is not None:
            load_weights(m, "piqo")
        return cls(model=m, resolver=resolver)


