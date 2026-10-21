import logging
from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable

from data.case import DecisionCase
from inference.options import IAnswer, IOption, option_to_candidates, probabilities_to_answer
from model.torch_model import IModel
import scipy.special as sp
import torch

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
        candidates = option_to_candidates(option)
        with torch.no_grad():
            scores = self.model.forward([DecisionCase(state=state, candidates=candidates, instruction=option.instruction)])[0]
        probs = sp.softmax(scores.tolist())
        answer = probabilities_to_answer(option, probs)
        logger.info("Decided %s with %.1f%% confidence across %d candidates", type(option).__name__, answer.confidence * 100, len(candidates))
        return answer

    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine":
        return layer(self)
