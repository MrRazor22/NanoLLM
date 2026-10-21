from typing import Optional
import httpx
from inference.options import (
    ChoiceAnswer, ChoiceOption, IAnswer, IOption, NoulAnswer, NoulOption,
    ScoreAnswer, ScoreOption,
)
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer

class KevEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Jared Palmer's Kev decision server (/v1/systemone)."""

    def __init__(
        self,
        endpoint: str = "",
        timeout: float = 30.0,
        device: Optional[str] = None,
        model: str = "kev-4b",
        api_key: str = "lm-studio",
    ):
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
        self.endpoint = endpoint or os.environ.get("KEV_ENDPOINT", "")
        self.timeout = timeout
        self.last_latency_ms: float = 0.0
        self.candidate_endpoints = [self.endpoint] if self.endpoint else [
            "http://127.0.0.1:1234/v1/systemone",
            "http://127.0.0.1:8008/v1/systemone",
        ]

    def decide(self, state: str, option: IOption) -> IAnswer:
        if isinstance(option, ChoiceOption):
            opts = option.options if isinstance(option.options, dict) else {o: o for o in option.options}
            q_spec = {"type": "choice", "criteria": opts}
            if option.instruction:
                q_spec["instructions"] = option.instruction
        elif isinstance(option, NoulOption):
            q_spec = {"type": "noul", "instructions": option.instruction}
            if option.criteria:
                q_spec["criteria"] = dict(option.criteria)
        elif isinstance(option, ScoreOption):
            q_spec = {"type": "score", "criteria": list(option.criteria)}
            if option.instruction:
                q_spec["instructions"] = option.instruction
        else:
            raise ValueError(f"Unsupported option type: {type(option)}")

        payload = {"model": self.model, "state": str(state), "questions": {"decision": q_spec}}
        raw, last_err = None, None
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for ep in self.candidate_endpoints:
            try:
                resp = httpx.post(ep, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                raw = resp.json()
                self.candidate_endpoints = [ep]
                break
            except Exception as e:
                last_err = e
                continue

        if raw is None:
            raise ConnectionError(
                f"Cannot connect to Kev decision server (tried {self.candidate_endpoints}): {last_err}. "
                "Ensure LM Studio local server (port 1234) or 'llama serve' (port 8008) is running."
            )

        choices = raw.get("choices", raw.get("answers", {}))
        nouls = raw.get("nouls", {})
        scores = raw.get("scores", {})

        if isinstance(option, ChoiceOption) and "decision" in choices:
            c_data = choices["decision"]
            choice_str = str(c_data.get("choice", ""))
            conf = float(c_data.get("confidence", 1.0))
            return ChoiceAnswer(choice=choice_str, confidence=conf, probabilities=c_data.get("probabilities", {choice_str: conf}))
        elif isinstance(option, NoulOption) and ("decision" in nouls or "decision" in choices):
            n_data = nouls.get("decision", choices.get("decision", {}))
            val = n_data.get("noul", n_data.get("decision"))
            p = float(n_data.get("probability", 1.0 if val else 0.0))
            noul_val = float(val) if isinstance(val, (int, float)) else (p if val else 1.0 - p)
            return NoulAnswer(noul=noul_val, decision=bool(val), confidence=p)
        elif isinstance(option, ScoreOption) and ("decision" in scores or "decision" in choices):
            s_data = scores.get("decision", choices.get("decision", {}))
            if "score" in s_data:
                score_val = float(s_data["score"])
                return ScoreAnswer(score=score_val, confidence=1.0)

        raise ValueError(f"Unsupported option type or missing decision in Kev response: {type(option)}")

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
