from typing import Callable, Sequence
from inference import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    IDecisionEngine, Noul, NoulResult, Question, Score, ScoreResult,
)

class GlinerDecideEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Fastino's GLiNER2.5-Decide."""

    def __init__(self, model_id: str = "fastino/GLiNER2.5-Decide"):
        self.model_id = model_id
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from gliner2 import AutoExtractor
                self._model = AutoExtractor.from_pretrained(self.model_id)
            except ImportError:
                try:
                    from gliner import GLiNER
                    self._model = GLiNER.from_pretrained(self.model_id)
                except ImportError as e:
                    raise ImportError(f"Cannot load GLiNER2.5-Decide ({self.model_id}): {e}. To run live: pip install gliner2")
        return self._model

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        model = self._get_model()
        answers = {}
        for q in questions:
            prompt_text = f"Question: {q.instruction}\n\nContext:\n{state}" if q.instruction else str(state)
            if isinstance(q, Choice):
                if isinstance(q.options, dict):
                    opt_map = {desc if desc else k: k for k, desc in q.options.items()}
                else:
                    opt_map = {opt: opt for opt in q.options}
                labels = list(opt_map.keys())
                if hasattr(model, "classify"):
                    preds = model.classify(prompt_text, labels)
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

__all__ = ["GlinerDecideEngine"]
