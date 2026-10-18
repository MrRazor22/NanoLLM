from typing import Optional
from data.case import (
    ChoiceAnswer, ChoiceOption, IOption, NoulAnswer, NoulOption, ScoreOption,
)
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
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
        self.last_latency_ms: float = 0.0

    def decide(self, state: str, option: IOption) -> IOption:
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
        prompt_text = str(state)

        if isinstance(option, ChoiceOption):
            opt_map = {f"It is {desc or k}": k for k, desc in option.options.items()} if isinstance(option.options, dict) else {f"It is {o}": o for o in option.options}
            res = pipe(prompt_text, list(opt_map.keys()))
            items = res[0] if (res and isinstance(res, list) and isinstance(res[0], list)) else (res or [])
            cand_scores = {opt_map[it["label"]]: float(it.get("score", 0.0)) for it in items if it.get("label") in opt_map}
            if cand_scores:
                total = sum(cand_scores.values()) or 1.0
                norm_p = {k: v / total for k, v in cand_scores.items()}
                top_k = opt_map.get(items[0].get("label", ""))
                ans = ChoiceAnswer(choice=str(top_k), confidence=norm_p.get(top_k, 1.0), probabilities=norm_p)
                return ChoiceOption(options=option.options, answer=ans)
        elif isinstance(option, NoulOption):
            res = pipe(prompt_text, ["It is true", "It is false"])
            items = res[0] if (res and isinstance(res, list) and isinstance(res[0], list)) else (res or [])
            s_map = {it["label"]: float(it.get("score", 0.0)) for it in items if "label" in it}
            if s_map:
                prob = s_map.get("It is true", 0.0) / ((s_map.get("It is true", 0.0) + s_map.get("It is false", 0.0)) or 1.0)
                ans = NoulAnswer(decision=prob >= 0.5, confidence=prob if prob >= 0.5 else 1.0 - prob)
                return NoulOption(answer=ans)
        elif isinstance(option, ScoreOption):
            raise NotImplementedError("Verdict does not support continuous regression decisions")

        return option

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
