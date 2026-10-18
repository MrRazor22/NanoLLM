import os
from typing import Optional
import httpx
from data.case import (
    ChoiceAnswer, ChoiceOption, IOption, NoulAnswer, NoulOption,
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

    def decide(self, state: str, option: IOption) -> IOption:
        if not self.api_key and "api.typesafe.ai" in self.endpoint:
            raise ValueError("TypeSafe Jev API key not set. Set JEV_API_KEY environment variable to evaluate hosted Jev.")

        if isinstance(option, ChoiceOption):
            opts = option.options if isinstance(option.options, dict) else {o: o for o in option.options}
            q_spec = {"type": "choice", "criteria": opts}
        elif isinstance(option, NoulOption):
            q_spec = {"type": "noul", "criteria": {"false": "No", "true": "Yes"}}
        else:
            q_spec = {"type": "score", "min": option.min_value, "max": option.max_value}

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
            ans = ChoiceAnswer(choice=str(choice), confidence=float(conf), probabilities=probs)
            return ChoiceOption(options=option.options, answer=ans)
        elif isinstance(option, NoulOption):
            is_pos = res.get("decision", False)
            conf = res.get("probability", 1.0 if is_pos else 0.0)
            ans = NoulAnswer(decision=bool(is_pos), confidence=float(conf))
            return NoulOption(answer=ans)
        elif isinstance(option, ScoreOption):
            val = float(res.get("score", 0.0))
            ans = ScoreAnswer(score=val, confidence=1.0)
            return ScoreOption(min_value=option.min_value, max_value=option.max_value, answer=ans)
        return option

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
