from pathlib import Path
from typing import Optional
import time
from inference.options import (
    ChoiceAnswer, ChoiceOption, IAnswer, IOption, NoulAnswer, NoulOption, ScoreAnswer, ScoreOption,
)
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from model.device import resolve_device

LOCAL_D1_PATH = Path(__file__).resolve().parents[2] / "spec" / "competitors" / "d1-omni-600M"

class D1OmniEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Liquid AI's d1-omni-600M multimodal decision model."""

    def __init__(
        self,
        model_id: str = "LiquidAI/d1-omni-600M",
        device: Optional[str] = None,
    ):
        self.model_path = str(LOCAL_D1_PATH) if LOCAL_D1_PATH.exists() else model_id
        self.device = resolve_device(device)
        self._model = None
        self.last_latency_ms: float = 0.0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        try:
            self._model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self.device)
        except Exception as e:
            raise ImportError(
                f"Cannot load d1-omni-600M ({self.model_path}): {e}. "
                "Ensure transformers>=5.15 and model weights are available."
            )

    def decide(self, state: str, option: IOption) -> IAnswer:
        self._ensure_loaded()
        t0 = time.perf_counter()

        if isinstance(option, ChoiceOption):
            crit = option.options if isinstance(option.options, dict) else {o: o for o in option.options}
            questions = {"decision": {"type": "choice", "instructions": "Select the correct option.", "criteria": crit}}
        elif isinstance(option, NoulOption):
            instructions = option.instruction or "Determine if the condition holds."
            questions = {"decision": {"type": "noul", "instructions": instructions}}
        elif isinstance(option, ScoreOption):
            questions = {"decision": {"type": "score", "instructions": "Rate this on the scale.", "criteria": list(option.criteria)}}
        else:
            raise ValueError(f"Unsupported option type: {type(option)}")

        results = self._model.system_one(state, questions)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        decision = results.get("decision", {})
        if isinstance(option, ChoiceOption):
            choice = decision.get("choice", "")
            probs = decision.get("probabilities", {})
            conf = decision.get("confidence", probs.get(choice, 1.0))
            return ChoiceAnswer(choice=str(choice), confidence=float(conf), probabilities=probs)
        elif isinstance(option, NoulOption):
            is_pos = decision.get("decision", False)
            conf = decision.get("probability", 1.0 if is_pos else 0.0)
            return NoulAnswer(noul=float(conf), decision=bool(is_pos), confidence=float(conf))
        elif isinstance(option, ScoreOption):
            val = float(decision.get("score", 0.0))
            return ScoreAnswer(score=val, confidence=1.0)

        raise ValueError(f"Unexpected outcome for option type: {type(option)}")

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
