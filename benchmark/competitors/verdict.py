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
                self._pipeline = ZeroShotClassificationPipeline(
                    model, tokenizer, classification_type="single_label", progress_bar=False
                )
            except Exception as e:
                raise ImportError(f"Cannot load Verdict ({self.model_id}): {e}. To run live: pip install gliclass transformers")
        return self._pipeline

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        pipe = self._get_pipeline()
        answers = {}
        for q in questions:
            prompt_text = f"Question: {q.instruction}\n\nContext:\n{state}" if q.instruction else str(state)
            if isinstance(q, Choice):
                if isinstance(q.options, dict):
                    opt_map = {f"It is {desc if desc else key}": key for key, desc in q.options.items()}
                else:
                    opt_map = {f"It is {opt}": opt for opt in q.options}
                candidates = list(opt_map.keys())
                out = pipe(prompt_text, candidates)
                res_list = out[0] if (out and isinstance(out, list) and isinstance(out[0], list)) else (out or [])
                top = res_list[0] if res_list else {}
                top_label = str(top.get("label", candidates[0] if candidates else ""))
                chosen_key = opt_map.get(top_label, top_label)
                score = float(top.get("score", 1.0))
                answers[q.name] = ChoiceResult(choice=str(chosen_key), confidence=score, probabilities={str(chosen_key): score})
            elif isinstance(q, Noul):
                nli_map = {"It is true": True, "It is false": False}
                out = pipe(prompt_text, list(nli_map.keys()))
                res_list = out[0] if (out and isinstance(out, list) and isinstance(out[0], list)) else (out or [])
                top = res_list[0] if res_list else {}
                top_label = str(top.get("label", "It is false"))
                is_pos = nli_map.get(top_label, False)
                score = float(top.get("score", 0.5))
                answers[q.name] = NoulResult(is_positive=is_pos, probability=score)
            elif isinstance(q, Score):
                answers[q.name] = ScoreResult(score=50.0, normalized=0.5)
        return DecisionResult(answers=answers)

    def __or__(self, layer: Callable[[IDecisionEngine], IDecisionEngine]) -> IDecisionEngine:
        return layer(self)

__all__ = ["VerdictEngine"]
