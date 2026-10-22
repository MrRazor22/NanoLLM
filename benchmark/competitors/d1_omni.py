from pathlib import Path
from typing import Optional

from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.options import (
    ChoiceAnswer, ChoiceOption, IAnswer, IOption, NoulAnswer, NoulOption, ScoreAnswer, ScoreOption,
)
from model.device import resolve_device

LOCAL_D1_PATH = Path(__file__).resolve().parents[2] / "spec" / "competitors" / "d1-omni-600M"

class D1OmniEngine(IDecisionEngine):
    def __init__(self, model_id: str = "LiquidAI/d1-omni-600M", device: Optional[str] = None):
        self.model_path = str(LOCAL_D1_PATH) if LOCAL_D1_PATH.exists() else model_id
        self.device = resolve_device(device)
        self._model = None

    def decide(self, state: str, option: IOption) -> IAnswer:
        if self._model is None:
            import torch
            from transformers import AutoModel
            dtype = torch.float32 if self.device.type == "cpu" else torch.float16
            self._model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True, torch_dtype=dtype
            ).to(self.device)

        match option:
            case ChoiceOption(criteria=crit, instruction=inst):
                query = {"type": "choice", "instructions": inst, "criteria": {str(k): str(v or k) for k, v in crit.items()}}
            case NoulOption(instruction=inst, criteria=crit):
                query = {"type": "noul", "instructions": inst, "criteria": {str(k): str(v or k) for k, v in crit.items()}}
            case ScoreOption(criteria=crit, instruction=inst):
                query = {"type": "score", "instructions": inst, "criteria": list(crit)}
            case _:
                raise ValueError(f"Unsupported option type: {type(option)}")

        results = self._model.system_one(state, {"decision": query})
        decision = results.get("answers", {}).get("decision", results.get("decision", {}))

        match option:
            case ChoiceOption():
                choice = str(decision.get("choice", ""))
                probs = decision.get("probabilities", {})
                return ChoiceAnswer(choice=choice, confidence=float(decision.get("confidence", probs.get(choice, 1.0))), probabilities=probs)
            case NoulOption():
                prob = float(decision.get("noul", decision.get("probability", 0.0)))
                return NoulAnswer(noul=prob, decision=prob >= 0.5, confidence=max(prob, 1.0 - prob))
            case ScoreOption():
                return ScoreAnswer(score=float(decision.get("score", 0.0)), confidence=float(decision.get("confidence", 1.0)))

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
