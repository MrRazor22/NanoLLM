import sys
from pathlib import Path
from typing import Optional

from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.options import (
    ChoiceAnswer, ChoiceOption, IAnswer, IOption, NoulAnswer, NoulOption, ScoreAnswer, ScoreOption,
)
from model.device import resolve_device

LOCAL_GLINER_DIR = Path(__file__).resolve().parents[2] / "spec" / "competitors" / "gliner"

class GlinerDecideEngine(IDecisionEngine):
    def __init__(self, model_id: str = "fastino/GLiNER2.5-Decide", device: Optional[str] = None):
        self.model_id = model_id
        self.device = resolve_device(device)
        self._model = None

    def decide(self, state: str, option: IOption) -> IAnswer:
        if self._model is None:
            if str(LOCAL_GLINER_DIR) not in sys.path and LOCAL_GLINER_DIR.exists():
                sys.path.insert(0, str(LOCAL_GLINER_DIR))
            from gliner2 import AutoExtractor
            self._model = AutoExtractor.from_pretrained(self.model_id).to(self.device)

        task = option.instruction or "decision"
        match option:
            case ChoiceOption(criteria=crit) | NoulOption(criteria=crit):
                labels = {str(k): str(v or k) for k, v in crit.items()}
            case ScoreOption(criteria=crit):
                labels = {str(i): str(level) for i, level in enumerate(crit)}
            case _:
                raise ValueError(f"Unsupported option type: {type(option)}")

        res = self._model.classify_text(str(state), {task: labels})
        winner = str(res.get(task, ""))

        match option:
            case ChoiceOption():
                return ChoiceAnswer(choice=winner, confidence=1.0, probabilities={k: (1.0 if k == winner else 0.0) for k in labels})
            case NoulOption():
                is_pos = (winner == "true")
                return NoulAnswer(noul=1.0 if is_pos else 0.0, decision=is_pos, confidence=1.0)
            case ScoreOption():
                val = float(winner) if winner.isdigit() else 0.0
                return ScoreAnswer(score=val, confidence=1.0)

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
