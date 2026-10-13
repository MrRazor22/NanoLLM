import json
from typing import Callable, Optional, Sequence
import urllib.request
from inference import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    IDecisionEngine, Noul, NoulResult, Question, Score, ScoreResult,
)

class KevEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Jared Palmer's Kev decision server (/v1/systemone)."""

    def __init__(self, endpoint: str = "", timeout: float = 30.0, device: Optional[str] = None):
        import os
        self.endpoint = endpoint or os.environ.get("KEV_ENDPOINT", "")
        self.timeout = timeout
        self.candidate_endpoints = [self.endpoint] if self.endpoint else [
            "http://127.0.0.1:8008/v1/systemone",
            "http://127.0.0.1:1234/v1/systemone",
        ]

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        q_dict = {}
        for q in questions:
            if isinstance(q, Choice):
                q_dict[q.name] = {
                    "type": "choice",
                    "instructions": q.instruction or "",
                    "criteria": q.options if isinstance(q.options, dict) else {o: o for o in q.options},
                }
            elif isinstance(q, Noul):
                q_dict[q.name] = {"type": "noul", "instructions": q.instruction or ""}
            elif isinstance(q, Score):
                q_dict[q.name] = {
                    "type": "score",
                    "instructions": q.instruction or "",
                    "criteria": [str(q.min_value), str(q.max_value)],
                }

        payload = json.dumps({"state": str(state), "questions": q_dict}).encode("utf-8")
        raw = None
        last_err = None
        for ep in self.candidate_endpoints:
            req = urllib.request.Request(
                ep,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
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

        answers = {}
        choices = raw.get("choices", raw.get("answers", {}))
        nouls = raw.get("nouls", {})
        scores = raw.get("scores", {})

        for q in questions:
            if isinstance(q, Choice):
                c_data = choices.get(q.name, {})
                choice_str = str(c_data.get("choice", ""))
                conf = float(c_data.get("confidence", 1.0))
                answers[q.name] = ChoiceResult(choice=choice_str, confidence=conf, probabilities={choice_str: conf})
            elif isinstance(q, Noul):
                n_data = nouls.get(q.name, choices.get(q.name, {}))
                val = n_data.get("noul", False)
                p = 1.0 if val else 0.0
                answers[q.name] = NoulResult(is_positive=bool(val), probability=p)
            elif isinstance(q, Score):
                s_data = scores.get(q.name, choices.get(q.name, {}))
                score_val = float(s_data.get("score", 50.0))
                answers[q.name] = ScoreResult(score=score_val, normalized=score_val / 100.0)

        return DecisionResult(answers=answers)

    def __or__(self, layer: Callable[[IDecisionEngine], IDecisionEngine]) -> IDecisionEngine:
        return layer(self)

__all__ = ["KevEngine"]
