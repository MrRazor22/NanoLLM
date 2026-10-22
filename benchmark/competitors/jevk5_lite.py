import sys
from pathlib import Path
from typing import Optional

from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.options import (
    ChoiceAnswer, ChoiceOption, IAnswer, IOption, NoulAnswer, NoulOption, ScoreAnswer, ScoreOption,
)
from model.device import resolve_device

LOCAL_JEVK5_DIR = Path(__file__).resolve().parents[2] / "spec" / "competitors" / "jevk5"

class JevK5LiteEngine(IDecisionEngine):
    def __init__(self, model_id: str = "alibiserikbay/JevK5-Lite", device: Optional[str] = None):
        self.model_id = model_id
        self.device = resolve_device(device)
        self._model = None

    def decide(self, state: str, option: IOption) -> IAnswer:
        if self._model is None:
            if str(LOCAL_JEVK5_DIR) not in sys.path and LOCAL_JEVK5_DIR.exists():
                sys.path.insert(0, str(LOCAL_JEVK5_DIR))
            from jevk5.lite import JevK5Lite
            self._model = JevK5Lite.from_pretrained(self.model_id)

        task_name = option.instruction or "decision"
        match option:
            case ChoiceOption(criteria=crit):
                pairs = [(str(k), str(v or k)) for k, v in crit.items()]
                labels = [text for _, text in pairs]
                if len(set(labels)) < len(labels):
                    labels = [f"{k}: {text}" for k, text in pairs]
                keys = [k for k, _ in pairs]
            case NoulOption(criteria=crit):
                true_label = str(crit.get("true", "true"))
                false_label = str(crit.get("false", "false"))
                labels = [true_label, false_label]
                keys = ["true", "false"]
            case ScoreOption(criteria=crit):
                pairs = [(str(i), str(level)) for i, level in enumerate(crit)]
                labels = [text for _, text in pairs]
                keys = [k for k, _ in pairs]
            case _:
                raise ValueError(f"Unsupported option type: {type(option)}")

        res = self._model.classify(state, {task_name: labels}).get(task_name, {})
        raw_probs = res.get("probabilities", {})
        probs = {key: float(raw_probs.get(label, 0.0)) for key, label in zip(keys, labels)}

        match option:
            case ChoiceOption():
                chosen = max(probs, key=probs.get) if probs else ""
                return ChoiceAnswer(choice=chosen, confidence=probs.get(chosen, 1.0), probabilities=probs)
            case NoulOption():
                prob = probs.get("true", 0.0)
                return NoulAnswer(noul=prob, decision=prob >= 0.5, confidence=max(prob, 1.0 - prob))
            case ScoreOption():
                val = sum(float(k) * p for k, p in probs.items() if k.isdigit())
                return ScoreAnswer(score=val, confidence=max(probs.values(), default=1.0))

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
