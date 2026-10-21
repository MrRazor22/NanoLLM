import os
from typing import Optional
import httpx
from inference.options import (
    ChoiceAnswer, ChoiceOption, IAnswer, IOption, NoulAnswer, NoulOption,
    ScoreAnswer, ScoreOption,
)
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer

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
        self.last_latency_ms: float = 0.0

    def decide(self, state: str, option: IOption) -> IAnswer:
        if not self.api_key and "api.typesafe.ai" in self.endpoint:
            raise ValueError("TypeSafe Jev API key not set. Set JEV_API_KEY environment variable to evaluate hosted Jev.")

        if isinstance(option, ChoiceOption):
            q_spec = {"type": "choice", "instruction": option.instruction, "criteria": dict(option.criteria)}
        elif isinstance(option, NoulOption):
            q_spec = {"type": "noul", "instruction": option.instruction, "criteria": dict(option.criteria)}
        elif isinstance(option, ScoreOption):
            q_spec = {"type": "score", "instruction": option.instruction, "criteria": list(option.criteria)}
        else:
            raise ValueError(f"Unsupported option type: {type(option)}")

        payload = {
            "model": self.model,
            "state": state,
            "questions": {"decision": q_spec},
        }

        resp = httpx.post(
            self.endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self.last_latency_ms = float(data.get("latency_ms", 0.0))

        res = data.get("results", {}).get("decision", {})
        if isinstance(option, ChoiceOption):
            choice = res.get("choice", "")
            probs = res.get("probabilities", {})
            conf = res.get("confidence", probs.get(choice, 1.0))
            return ChoiceAnswer(choice=str(choice), confidence=float(conf), probabilities=probs)
        elif isinstance(option, NoulOption):
            is_pos = res.get("decision", False)
            conf = res.get("probability", 1.0 if is_pos else 0.0)
            noul_prob = float(res.get("noul", conf if is_pos else 1.0 - conf))
            return NoulAnswer(noul=noul_prob, decision=bool(is_pos), confidence=float(conf))
        elif isinstance(option, ScoreOption):
            val = float(res.get("score", 0.0))
            return ScoreAnswer(score=val, confidence=1.0)
        raise ValueError(f"Unsupported option type: {type(option)}")

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
