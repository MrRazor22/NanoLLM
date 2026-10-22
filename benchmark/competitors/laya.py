import sys
from pathlib import Path
from typing import Optional

from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.options import (
    ChoiceAnswer, ChoiceOption, IAnswer, IOption, NoulAnswer, NoulOption,
    ScoreAnswer, ScoreOption,
)
from model.device import resolve_device

LOCAL_LAYA_DIR = Path(__file__).resolve().parents[2] / "spec" / "competitors" / "laya"

class LayaEngine(IDecisionEngine):
    def __init__(self, model_variant: str = "convaiinnovations/laya", device: Optional[str] = None):
        self.model_variant = model_variant
        self.device = resolve_device(device)
        self._model = None

    def decide(self, state: str, option: IOption) -> IAnswer:
        if self._model is None:
            if str(LOCAL_LAYA_DIR) not in sys.path and LOCAL_LAYA_DIR.exists():
                sys.path.insert(0, str(LOCAL_LAYA_DIR))
            from laya import Agent
            self._model = Agent(self.model_variant, device=self.device)

        match option:
            case ChoiceOption(instruction=inst, criteria=crit):
                spec = {"type": "choice", "instructions": inst, "criteria": {str(k): str(v or k) for k, v in crit.items()}}
            case NoulOption(instruction=inst, criteria=crit):
                spec = {"type": "noul", "instructions": inst, "criteria": {str(k): str(v or k) for k, v in crit.items()}}
            case ScoreOption(instruction=inst, criteria=crit):
                spec = {"type": "score", "instructions": inst, "criteria": list(crit)}
            case _:
                raise ValueError(f"Unsupported option type: {type(option)}")

        results = self._model.system_one(str(state), {"decision": spec})
        raw = results.get("answers", {}).get("decision", results.get("decision", {}))

        match option:
            case ChoiceOption():
                choice = str(raw.get("choice", ""))
                conf = float(raw.get("confidence", 1.0))
                return ChoiceAnswer(choice=choice, confidence=conf, probabilities=raw.get("probabilities", {choice: conf}))
            case NoulOption():
                prob = float(raw.get("noul", raw.get("probability", 0.0)))
                return NoulAnswer(noul=prob, decision=prob >= 0.5, confidence=max(prob, 1.0 - prob))
            case ScoreOption():
                return ScoreAnswer(score=float(raw.get("score", 0.0)), confidence=float(raw.get("confidence", 1.0)))

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
