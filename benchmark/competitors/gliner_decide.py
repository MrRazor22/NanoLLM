from typing import Optional
from data.case import (
    ChoiceAnswer, ChoiceOption, IOption, NoulAnswer, NoulOption, ScoreOption,
)
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from model.device import resolve_device

class GlinerDecideEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Fastino's GLiNER2.5-Decide."""

    def __init__(
        self,
        model_id: str = "fastino/GLiNER2.5-Decide",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = resolve_device(device)
        self._model = None
        self.last_latency_ms: float = 0.0

    def decide(self, state: str, option: IOption) -> IOption:
        if self._model is None:
            try:
                try:
                    from gliner2 import AutoExtractor
                    m = AutoExtractor.from_pretrained(self.model_id)
                except (ImportError, TypeError):
                    from gliner import GLiNER
                    m = GLiNER.from_pretrained(self.model_id)
                if hasattr(m, "to"): m.to(self.device)
                self._model = m
            except Exception as e:
                raise ImportError(f"Cannot load GLiNER2.5-Decide ({self.model_id}): {e}. To run live: pip install gliner2")

        model = self._model
        prompt_text = str(state)

        if isinstance(option, ChoiceOption):
            opt_map = {desc or k: k for k, desc in option.options.items()} if isinstance(option.options, dict) else {opt: opt for opt in option.options}
            if hasattr(model, "classify_text"):
                raw = model.classify_text(prompt_text, {"decision": list(opt_map.keys())}).get("decision", "")
                if raw in opt_map:
                    k = opt_map[raw]
                    ans = ChoiceAnswer(choice=str(k), confidence=1.0, probabilities={v: (1.0 if v == k else 0.0) for v in opt_map.values()})
                    return ChoiceOption(options=option.options, answer=ans)
            else:
                preds = model.classify(prompt_text, list(opt_map.keys())) if hasattr(model, "classify") else model.predict_entities(prompt_text, list(opt_map.keys()))
                scores = {opt_map[p["label"]]: float(p.get("score", 0.0)) for p in (preds or []) if p.get("label") in opt_map}
                if scores and preds and preds[0].get("label") in opt_map:
                    total = sum(scores.values()) or 1.0
                    norm_p = {k: v / total for k, v in scores.items()}
                    top_k = opt_map[preds[0]["label"]]
                    ans = ChoiceAnswer(choice=str(top_k), confidence=norm_p.get(top_k, 1.0), probabilities=norm_p)
                    return ChoiceOption(options=option.options, answer=ans)
        elif isinstance(option, NoulOption):
            if hasattr(model, "classify_text"):
                ans_label = model.classify_text(prompt_text, {"decision": ["Yes", "No"]}).get("decision", "")
                if ans_label in ("Yes", "No"):
                    is_pos = (ans_label == "Yes")
                    ans = NoulAnswer(decision=is_pos, confidence=1.0)
                    return NoulOption(answer=ans)
            else:
                preds = model.classify(prompt_text, ["Yes", "No"]) if hasattr(model, "classify") else model.predict_entities(prompt_text, ["Yes", "No"])
                if preds and preds[0].get("label") in ("Yes", "No"):
                    is_pos = (preds[0]["label"] == "Yes")
                    score = float(preds[0].get("score", 0.5))
                    ans = NoulAnswer(decision=is_pos, confidence=score if is_pos else 1.0 - score)
                    return NoulOption(answer=ans)
        elif isinstance(option, ScoreOption):
            raise NotImplementedError("GLiNER does not support continuous regression decisions")
        return option

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
