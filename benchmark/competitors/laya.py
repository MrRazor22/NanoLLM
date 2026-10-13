from typing import Callable, Optional, Sequence
from inference import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    IDecisionEngine, Noul, NoulResult, Question, Score, ScoreResult,
)
from model import resolve_device

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

    def _get_agent(self):
        if self._agent is None:
            try:
                from laya import Agent
                self._agent = Agent(self.model_variant, device=self.device)
            except ImportError:
                raise ImportError("Laya is not installed. To run live: pip install laya")
        return self._agent

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        agent = self._get_agent()
        q_dict = {}
        for q in questions:
            instr = q.instruction.strip() if q.instruction and q.instruction.strip() else f"Select the best option for {q.name}."
            if isinstance(q, Choice):
                q_dict[q.name] = {
                    "type": "choice",
                    "instructions": instr,
                    "criteria": q.options if isinstance(q.options, dict) else {o: o for o in q.options},
                }
            elif isinstance(q, Noul):
                q_dict[q.name] = {"type": "noul", "instructions": instr}
            elif isinstance(q, Score):
                q_dict[q.name] = {"type": "score", "instructions": instr, "criteria": [str(q.min_value), str(q.max_value)]}

        raw = agent.system_one(str(state), q_dict)
        answers = {}
        for q in questions:
            ans_raw = raw.get("answers", {}).get(q.name, {})
            if isinstance(q, Choice):
                answers[q.name] = ChoiceResult(choice=str(ans_raw.get("choice", "")), confidence=float(ans_raw.get("confidence", 1.0)), probabilities={})
            elif isinstance(q, Noul):
                p = float(ans_raw.get("noul", 0.0))
                answers[q.name] = NoulResult(is_positive=p >= 0.5, probability=p)
            elif isinstance(q, Score):
                s = float(ans_raw.get("score", 0.0))
                answers[q.name] = ScoreResult(score=s, normalized=s / 100.0)

        return DecisionResult(answers=answers)

    def __or__(self, layer: Callable[[IDecisionEngine], IDecisionEngine]) -> IDecisionEngine:
        return layer(self)

__all__ = ["LayaEngine"]
