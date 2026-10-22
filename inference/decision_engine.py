import logging
from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable
import scipy.special as sp

from data.case import DecisionCase
from inference.options import (
    ChoiceAnswer, ChoiceOption,
    IAnswer, IOption,
    NoulAnswer, NoulOption,
    ScoreAnswer, ScoreOption,
)
from model.torch_model import IModel

logger = logging.getLogger(__name__)

@runtime_checkable
class IDecisionEngineLayer(Protocol):
    def __call__(self, inner: "IDecisionEngine") -> "IDecisionEngine": ...

@runtime_checkable
class IDecisionEngine(Protocol):
    def decide(self, state: str, option: IOption) -> IAnswer: ...
    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine": ...

class DecisionEngine(IDecisionEngine):
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

    def decide(self, state: str, option: IOption) -> IAnswer:
        match option:
            case ChoiceOption(instruction=inst, criteria=crit):
                candidates = tuple(f"{k}: {v}" if v else k for k, v in crit.items())
            case NoulOption(instruction=inst, criteria=crit):
                candidates = (crit["false"], crit["true"])
            case ScoreOption(instruction=inst, criteria=crit):
                candidates = tuple(crit)
            case _:
                raise ValueError(f"Unsupported option: {type(option)}")

        scores = self.model.score([DecisionCase(state=state, candidates=candidates, instruction=inst)])[0]
        probs = sp.softmax(scores.tolist())

        match option:
            case ChoiceOption(criteria=crit):
                best = max(range(len(probs)), key=probs.__getitem__)
                answer = ChoiceAnswer(choice=tuple(crit)[best], confidence=float(probs[best]), probabilities=dict(zip(crit, map(float, probs))))
            case NoulOption():
                p_true = float(probs[1])
                answer = NoulAnswer(noul=p_true, decision=p_true >= 0.5, confidence=max(p_true, 1.0 - p_true))
            case ScoreOption():
                answer = ScoreAnswer(score=float(sum(i * p for i, p in enumerate(probs))), confidence=float(max(probs)), probabilities=list(map(float, probs)))

        logger.info("Decided %s with %.1f%% confidence across %d candidates", type(option).__name__, answer.confidence * 100, len(candidates))
        return answer

    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine":
        return layer(self)
