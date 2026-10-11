from typing import Callable, Sequence
from inference import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    IDecisionEngine, Noul, NoulResult, Question, Score, ScoreResult,
)

class VerdictEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Verdict 2.0 (heman10x/rlcd-modernbert-151m)."""

    def __init__(self, model_id: str = "heman10x/rlcd-modernbert-151m"):
        self.model_id = model_id
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline("text-classification", model=self.model_id)
            except Exception as e:
                raise ImportError(f"Cannot load Verdict ({self.model_id}): {e}. To run live: pip install transformers accelerate")
        return self._pipeline

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        pipe = self._get_pipeline()
        answers = {}
        prompt_text = str(state)
        for q in questions:
            if isinstance(q, Choice):
                options = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                out = pipe(f"{prompt_text} \nQuestion: {q.name}")
                top = out[0] if isinstance(out, list) else out
                label = str(top.get("label", options[0] if options else ""))
                score = float(top.get("score", 1.0))
                answers[q.name] = ChoiceResult(choice=label, confidence=score, probabilities={label: score})
            elif isinstance(q, Noul):
                out = pipe(f"{prompt_text} \nQuestion: {q.name}")
                top = out[0] if isinstance(out, list) else out
                score = float(top.get("score", 0.5))
                answers[q.name] = NoulResult(is_positive=score >= 0.5, probability=score)
            elif isinstance(q, Score):
                out = pipe(f"{prompt_text} \nQuestion: {q.name}")
                top = out[0] if isinstance(out, list) else out
                score = float(top.get("score", 0.5)) * 100.0
                answers[q.name] = ScoreResult(score=score, normalized=score / 100.0)
        return DecisionResult(answers=answers)

    def __or__(self, layer: Callable[[IDecisionEngine], IDecisionEngine]) -> IDecisionEngine:
        return layer(self)

__all__ = ["VerdictEngine"]
