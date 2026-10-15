import json
import os
from typing import Callable, Optional, Sequence
import httpx
from inference.decision_engine import IDecisionEngine
from inference.questions import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    Noul, NoulResult, Question, Score, ScoreResult,
)

class JevEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps TypeSafe AI's hosted Jev API (/v1/systemone)."""

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        model: str = "jev-1.13.0",
        timeout: float = 30.0,
        device: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("JEV_API_KEY", "")
        self.endpoint = endpoint or os.environ.get("JEV_ENDPOINT", "https://api.typesafe.ai/v1/systemone")
        self.timeout = timeout

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        if not self.api_key and "api.typesafe.ai" in self.endpoint:
            raise ValueError("TypeSafe Jev API key not set. Set JEV_API_KEY environment variable to evaluate hosted Jev.")

        q_dict = {}
        for q in questions:
            if isinstance(q, Choice):
                q_dict[q.name] = {
                    "type": "choice",
                    "instructions": q.instruction or "",
                    "criteria": q.options if isinstance(q.options, dict) else {o: o for o in q.options},
                }
            elif isinstance(q, Noul):
                q_dict[q.name] = {
                    "type": "noul",
                    "instructions": q.instruction or "",
                    "criteria": {"false": "No", "true": "Yes"},
                }
            elif isinstance(q, Score):
                q_dict[q.name] = {
                    "type": "score",
                    "instructions": q.instruction or "",
                    "min": q.min_value,
                    "max": q.max_value,
                }

        payload = {
            "model": self.model,
            "state": state,
            "questions": q_dict,
        }

        resp = httpx.post(
            self.endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        answers = {}
        results = data.get("results", {})
        for q in questions:
            res = results.get(q.name, {})
            if isinstance(q, Choice):
                choice = res.get("choice", "")
                probs = res.get("probabilities", {})
                conf = res.get("confidence", probs.get(choice, 1.0))
                answers[q.name] = ChoiceResult(choice=str(choice), confidence=float(conf), probabilities=probs)
            elif isinstance(q, Noul):
                is_pos = res.get("decision", False)
                p = res.get("probability", 1.0 if is_pos else 0.0)
                answers[q.name] = NoulResult(is_positive=bool(is_pos), probability=float(p))
            elif isinstance(q, Score):
                val = res.get("score", 0.0)
                answers[q.name] = ScoreResult(score=float(val), normalized=float(val) / 100.0)

        latency = float(data.get("latency_ms", 0.0))
        return DecisionResult(answers=answers, latency_ms=latency)

    def __or__(self, layer: Callable[[IDecisionEngine], IDecisionEngine]) -> IDecisionEngine:
        return layer(self)
