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
                from gliclass import GLiClassModel, ZeroShotClassificationPipeline
                from transformers import AutoTokenizer
                model = GLiClassModel.from_pretrained(self.model_id)
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type="single_label")
            except Exception as e:
                raise ImportError(f"Cannot load Verdict ({self.model_id}): {e}. To run live: pip install gliclass transformers")
        return self._pipeline

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        pipe = self._get_pipeline()
        answers = {}
        prompt_text = str(state)
        for q in questions:
            if isinstance(q, Choice):
                options = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                out = pipe(prompt_text, options)
                res_list = out[0] if (out and isinstance(out, list) and isinstance(out[0], list)) else (out or [])
                top = res_list[0] if res_list else {}
                label = str(top.get("label", options[0] if options else ""))
                score = float(top.get("score", 1.0))
                answers[q.name] = ChoiceResult(choice=label, confidence=score, probabilities={label: score})
            elif isinstance(q, Noul):
                out = pipe(prompt_text, ["true", "false"])
                res_list = out[0] if (out and isinstance(out, list) and isinstance(out[0], list)) else (out or [])
                top = res_list[0] if res_list else {}
                is_pos = str(top.get("label", "")).lower() in ("true", "1", "yes")
                score = float(top.get("score", 0.5))
                answers[q.name] = NoulResult(is_positive=is_pos, probability=score)
            elif isinstance(q, Score):
                answers[q.name] = ScoreResult(score=50.0, normalized=0.5)
        return DecisionResult(answers=answers)

    def __or__(self, layer: Callable[[IDecisionEngine], IDecisionEngine]) -> IDecisionEngine:
        return layer(self)

__all__ = ["VerdictEngine"]
