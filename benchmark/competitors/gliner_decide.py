from typing import Callable, Optional, Sequence
from inference.decision_engine import IDecisionEngine
from inference.questions import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    Noul, NoulResult, Question, Score, ScoreResult,
)
from model.model_extensions import resolve_device

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
                if isinstance(q.options, dict):
                    opt_map = {desc if desc else k: k for k, desc in q.options.items()}
                else:
                    opt_map = {opt: opt for opt in q.options}
                labels = list(opt_map.keys())

                if hasattr(model, "classify_text"):
                    preds = model.classify_text(prompt_text, {q.name: labels})
                    raw_choice = preds.get(q.name, "")
                    chosen_key = opt_map.get(raw_choice, raw_choice)
                    score = 1.0
                elif hasattr(model, "classify"):
                    preds = model.classify(prompt_text, labels)
                    top_label = preds[0]["label"] if preds else ""
                    score = float(preds[0].get("score", 0.0)) if preds else 0.0
                    chosen_key = opt_map.get(top_label, top_label)
                else:
                    preds = model.predict_entities(prompt_text, labels)
                    top_label = preds[0]["label"] if preds else ""
                    score = float(preds[0].get("score", 0.0)) if preds else 0.0
                    chosen_key = opt_map.get(top_label, top_label)

                answers[q.name] = ChoiceResult(choice=str(chosen_key), confidence=score, probabilities={str(chosen_key): score})
            elif isinstance(q, Noul):
                answers[q.name] = NoulResult(is_positive=False, probability=0.5)
            elif isinstance(q, Score):
                answers[q.name] = ScoreResult(score=50.0, normalized=0.5)
        return DecisionResult(answers=answers)

    def __or__(self, layer: Callable[[IDecisionEngine], IDecisionEngine]) -> IDecisionEngine:
        return layer(self)
