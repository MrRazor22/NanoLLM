import sys
from pathlib import Path
from typing import Optional
import time
from inference.options import (
    ChoiceAnswer, ChoiceOption, IAnswer, IOption, NoulAnswer, NoulOption, ScoreAnswer, ScoreOption,
)
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from model.device import resolve_device

LOCAL_JEVK5_DIR = Path(__file__).resolve().parents[2] / "spec" / "competitors" / "jevk5"

class JevK5LiteEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps JevK5-Lite (DeBERTa-v3-large CPU/GPU decision classifier)."""

    def __init__(
        self,
        model_id: str = "alibiserikbay/JevK5-Lite",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = resolve_device(device)
        self._lite = None
        self.last_latency_ms: float = 0.0

    def _ensure_loaded(self) -> None:
        if self._lite is not None:
            return
        if str(LOCAL_JEVK5_DIR) not in sys.path and LOCAL_JEVK5_DIR.exists():
            sys.path.insert(0, str(LOCAL_JEVK5_DIR))
        try:
            from jevk5.lite import JevK5Lite
            self._lite = JevK5Lite.from_pretrained(self.model_id)
        except Exception as e:
            raise ImportError(
                f"Cannot load JevK5-Lite ({self.model_id}): {e}. "
                "Ensure jevk5 and its dependencies are installed."
            )

    def decide(self, state: str, option: IOption) -> IAnswer:
        self._ensure_loaded()
        t0 = time.perf_counter()

        if isinstance(option, ChoiceOption):
            crit = option.options if isinstance(option.options, dict) else {o: o for o in option.options}
            labels = list(crit.keys())
            tasks = {"decision": labels}
        elif isinstance(option, NoulOption):
            labels = ["true", "false"]
            tasks = {"decision": labels}
        elif isinstance(option, ScoreOption):
            labels = [str(i) for i in range(len(option.criteria))]
            tasks = {"decision": labels}
        else:
            raise ValueError(f"Unsupported option type: {type(option)}")

        res = self._lite.classify(state, tasks)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        decision_data = res.get("decision", {})
        probs = decision_data.get("probabilities", {})

        if isinstance(option, ChoiceOption):
            chosen = decision_data.get("labels", [""])[0] if decision_data.get("labels") else ""
            conf = probs.get(chosen, 1.0)
            return ChoiceAnswer(choice=str(chosen), confidence=float(conf), probabilities=probs)
        elif isinstance(option, NoulOption):
            prob = float(probs.get("true", 0.0))
            is_pos = prob >= 0.5
            return NoulAnswer(noul=prob, decision=is_pos, confidence=prob if is_pos else 1.0 - prob)
        elif isinstance(option, ScoreOption):
            # Expectation over ordinal indices
            val = sum(float(i) * float(p) for i, p in probs.items() if i.isdigit())
            return ScoreAnswer(score=val, confidence=1.0)

        raise ValueError(f"Unexpected option type: {type(option)}")

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
