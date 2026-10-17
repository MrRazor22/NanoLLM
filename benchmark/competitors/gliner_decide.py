from typing import Optional, Sequence
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.questions import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    Noul, NoulResult, Question, Score, ScoreResult,
)
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

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
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
        answers = {}
        for q in questions:
            prompt_text = f"Question: {q.instruction}\n\nContext:\n{state}" if q.instruction else str(state)

            if isinstance(q, Choice):
                opt_map = {desc or k: k for k, desc in q.options.items()} if isinstance(q.options, dict) else {opt: opt for opt in q.options}
                if hasattr(model, "classify_text"):
                    raw = model.classify_text(prompt_text, {q.name: list(opt_map.keys())}).get(q.name, "")
                    if raw in opt_map:
                        k = opt_map[raw]
                        answers[q.name] = ChoiceResult(choice=str(k), confidence=1.0, probabilities={v: (1.0 if v == k else 0.0) for v in opt_map.values()})
                else:
                    preds = model.classify(prompt_text, list(opt_map.keys())) if hasattr(model, "classify") else model.predict_entities(prompt_text, list(opt_map.keys()))
                    scores = {opt_map[p["label"]]: float(p.get("score", 0.0)) for p in (preds or []) if p.get("label") in opt_map}
                    if scores and preds and preds[0].get("label") in opt_map:
                        total = sum(scores.values()) or 1.0
                        norm_p = {k: v / total for k, v in scores.items()}
                        top_k = opt_map[preds[0]["label"]]
                        answers[q.name] = ChoiceResult(choice=str(top_k), confidence=norm_p.get(top_k, 1.0), probabilities=norm_p)
            elif isinstance(q, Noul):
                if hasattr(model, "classify_text"):
                    ans_label = model.classify_text(prompt_text, {q.name: ["Yes", "No"]}).get(q.name, "")
                    if ans_label in ("Yes", "No"):
                        answers[q.name] = NoulResult(is_positive=(ans_label == "Yes"), probability=1.0 if ans_label == "Yes" else 0.0)
                else:
                    preds = model.classify(prompt_text, ["Yes", "No"]) if hasattr(model, "classify") else model.predict_entities(prompt_text, ["Yes", "No"])
                    if preds and preds[0].get("label") in ("Yes", "No"):
                        is_pos = (preds[0]["label"] == "Yes")
                        score = float(preds[0].get("score", 0.5))
                        answers[q.name] = NoulResult(is_positive=is_pos, probability=score if is_pos else 1.0 - score)
            elif isinstance(q, Score):
                raise NotImplementedError(f"GLiNER does not support continuous regression questions ({type(q).__name__})")
        return DecisionResult(answers=answers)

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
