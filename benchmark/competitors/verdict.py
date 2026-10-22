import sys
from pathlib import Path
from typing import Optional

from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.options import (
    ChoiceAnswer, ChoiceOption, IAnswer, IOption, NoulAnswer, NoulOption, ScoreAnswer, ScoreOption,
)
from model.device import resolve_device

LOCAL_VERDICT_DIR = Path(__file__).resolve().parents[2] / "spec" / "competitors" / "openJev-verdict-2.0"

class VerdictEngine(IDecisionEngine):
    def __init__(self, model_id: str = "heman10x/rlcd-modernbert-151m", device: Optional[str] = None):
        self.model_id = model_id
        self.device = resolve_device(device)
        self._model = None

    def decide(self, state: str, option: IOption) -> IAnswer:
        if self._model is None:
            if str(LOCAL_VERDICT_DIR) not in sys.path and LOCAL_VERDICT_DIR.exists():
                sys.path.insert(0, str(LOCAL_VERDICT_DIR))
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            from transformers import AutoTokenizer
            model = GLiClassModel.from_pretrained(self.model_id)
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = ZeroShotClassificationPipeline(
                model, tokenizer, classification_type="single_label", device=self.device, progress_bar=False
            )

        inst = option.instruction
        text = f"Question: {inst}\n\nContext:\n{state}" if inst else str(state)

        match option:
            case ChoiceOption(criteria=crit):
                pairs = [(str(k), f"{k}: {desc or k}") for k, desc in crit.items()]
            case NoulOption(criteria=crit):
                pairs = [("true", f"Proposition is true: {inst}" if inst else str(crit.get("true", "True"))), ("false", f"Proposition is false: {inst}" if inst else str(crit.get("false", "False")))]
            case ScoreOption(criteria=crit):
                pairs = [(str(i), f"Score {i}: {level}") for i, level in enumerate(crit)]
            case _:
                raise ValueError(f"Unsupported option type: {type(option)}")

        label_map = {label_text: key for key, label_text in pairs}
        res = self._model(text, list(label_map.keys()))
        items = res[0] if (res and isinstance(res, list) and isinstance(res[0], list)) else (res or [])
        raw_scores = {label_map[it["label"]]: float(it.get("score", 0.0)) for it in items if it.get("label") in label_map}
        if not raw_scores:
            raise ValueError(f"No prediction from Verdict for {type(option)}")

        total = sum(raw_scores.values()) or 1.0
        probs = {k: v / total for k, v in raw_scores.items()}

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
