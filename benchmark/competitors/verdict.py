from typing import Optional, Sequence
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.questions import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    Noul, NoulResult, Question, Score, ScoreResult,
)
from model.device import resolve_device

class VerdictEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Verdict 2.0 (heman10x/rlcd-modernbert-151m)."""

    def __init__(
        self,
        model_id: str = "heman10x/rlcd-modernbert-151m",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = resolve_device(device)
        self._pipeline = None

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        if self._pipeline is None:
            try:
                from gliclass import GLiClassModel, ZeroShotClassificationPipeline
                from transformers import AutoTokenizer
                model = GLiClassModel.from_pretrained(self.model_id)
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._pipeline = ZeroShotClassificationPipeline(
                    model, tokenizer, classification_type="single_label", device=self.device, progress_bar=False
                )
            except Exception as e:
                raise ImportError(f"Cannot load Verdict ({self.model_id}): {e}. To run live: pip install gliclass transformers")

        pipe = self._pipeline
        answers = {}
        for q in questions:
            prompt_text = f"Question: {q.instruction}\n\nContext:\n{state}" if q.instruction else str(state)
            if isinstance(q, Choice):
                opt_map = {f"It is {desc or k}": k for k, desc in q.options.items()} if isinstance(q.options, dict) else {f"It is {o}": o for o in q.options}
                res = pipe(prompt_text, list(opt_map.keys()))
                items = res[0] if (res and isinstance(res, list) and isinstance(res[0], list)) else (res or [])
                cand_scores = {opt_map[it["label"]]: float(it.get("score", 0.0)) for it in items if it.get("label") in opt_map}
                if not cand_scores: continue
                total = sum(cand_scores.values()) or 1.0
                norm_p = {k: v / total for k, v in cand_scores.items()}
                top_k = opt_map.get(items[0].get("label", ""))
                answers[q.name] = ChoiceResult(choice=str(top_k), confidence=norm_p.get(top_k, 1.0), probabilities=norm_p)
            elif isinstance(q, Noul):
                res = pipe(prompt_text, ["It is true", "It is false"])
                items = res[0] if (res and isinstance(res, list) and isinstance(res[0], list)) else (res or [])
                s_map = {it["label"]: float(it.get("score", 0.0)) for it in items if "label" in it}
                if not s_map: continue
                prob = s_map.get("It is true", 0.0) / ((s_map.get("It is true", 0.0) + s_map.get("It is false", 0.0)) or 1.0)
                answers[q.name] = NoulResult(is_positive=prob >= 0.5, probability=prob)
            elif isinstance(q, Score):
                raise NotImplementedError(f"Verdict does not support continuous regression questions ({type(q).__name__})")
        return DecisionResult(answers=answers)

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
