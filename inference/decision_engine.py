from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable

from inference.decision_resolver import DecisionResolver, IResolver
from inference.options import ChoiceOption, IOption, NoulOption, ScoreOption
from model.torch_model import IModel

@runtime_checkable
class IDecisionEngineLayer(Protocol):
    """Layer Decorator Contract (lambda: P -> P): Instruments or wraps decision engine execution."""
    def __call__(self, inner: "IDecisionEngine") -> "IDecisionEngine": ...

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Sovereign operational boundary for domain decision resolution."""
    def decide(self, state: str, option: IOption) -> IOption: ...
    def predict_candidates(self, state: str, candidates: Sequence[str]) -> Sequence[float]: ...
    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine": ...

def option_to_candidates(option: IOption) -> Sequence[str]:
    """Pure domain mapper: translates high-level ergonomic IOption to model candidate list."""
    match option:
        case ChoiceOption(options=opts):
            return tuple(opts.keys()) if isinstance(opts, dict) else tuple(opts)
        case ScoreOption(criteria=crit):
            return tuple(crit) if crit else ("0", "1", "2")
        case NoulOption():
            return ("false: does not hold", "true: holds")
        case _:
            raise ValueError(f"Unsupported option type: {type(option)}")

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

    def predict_candidates(self, state: str, candidates: Sequence[str]) -> Sequence[float]:
        pred = self.model.predict(state, candidates)
        import scipy.special as sp
        scores = [pred.scores[i] for i in pred.slots[0]]
        return list(sp.softmax(scores))

    def decide(self, state: str, option: IOption) -> IOption:
        candidates = option_to_candidates(option)
        pred = self.model.predict(state, candidates)
        return self.resolver.resolve(option, pred.slots, pred.scores)

    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine":
        return layer(self)
