from typing import Optional
from inference.options import (
    ChoiceAnswer, ChoiceOption, IOption, NoulAnswer, NoulOption,
    ScoreAnswer, ScoreOption,
)
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from model.device import resolve_device

class LayaEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Convai Innovations' Laya decision router."""

    def __init__(
        self,
        model_variant: str = "convaiinnovations/laya",
        device: Optional[str] = None,
    ):
        self.model_variant = model_variant
        self.device = resolve_device(device)
        self._agent = None
        self.last_latency_ms: float = 0.0

    def decide(self, state: str, option: IOption) -> IOption:
        if self._agent is None:
            try:
                from laya import Agent
                self._agent = Agent(self.model_variant, device=self.device)
            except ImportError:
                raise ImportError("Laya is not installed. To run live: pip install laya")

        if isinstance(option, ChoiceOption):
            opts = option.options if isinstance(option.options, dict) else {o: o for o in option.options}
            q_spec = {"type": "choice", "instructions": "Select the best option.", "criteria": opts}
        elif isinstance(option, NoulOption):
            q_spec = {"type": "noul", "instructions": "Make a binary decision."}
        else:
            q_spec = {"type": "score", "instructions": "Assign a score.", "criteria": [str(option.min_value), str(option.max_value)]}

        raw = self._agent.system_one(str(state), {"decision": q_spec})
        raw_answers = raw.get("answers", {})

        if isinstance(option, ChoiceOption) and "decision" in raw_answers:
            ans_raw = raw_answers["decision"]
            choice_str = str(ans_raw.get("choice", ""))
            conf = float(ans_raw.get("confidence", 1.0))
            probs = ans_raw.get("probabilities", {choice_str: conf})
            ans = ChoiceAnswer(choice=choice_str, confidence=conf, probabilities=probs)
            return ChoiceOption(options=option.options, answer=ans)
        elif isinstance(option, NoulOption) and "decision" in raw_answers:
            ans_raw = raw_answers["decision"]
            p = float(ans_raw.get("noul", ans_raw.get("probability", 0.0)))
            ans = NoulAnswer(decision=p >= 0.5, confidence=p if p >= 0.5 else 1.0 - p)
            return NoulOption(answer=ans)
        elif isinstance(option, ScoreOption) and "decision" in raw_answers:
            ans_raw = raw_answers["decision"]
            if "score" in ans_raw:
                s = float(ans_raw["score"])
                ans = ScoreAnswer(score=s, confidence=1.0)
                return ScoreOption(min_value=option.min_value, max_value=option.max_value, answer=ans)

        return option

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
