import time
from typing import TYPE_CHECKING, Optional, Sequence
import torch
from inference.questions import DecisionResult, DecisionState, Question

if TYPE_CHECKING:
    from inference.decision_engine import IDecisionEngine

class ProfilingLayer:
    """Layer Decorator (lambda: P -> P): Measures wall-clock decision latency."""
    def __init__(self, inner: Optional["IDecisionEngine"] = None):
        self.inner = inner

    def attach(self, inner: "IDecisionEngine") -> "ProfilingLayer":
        self.inner = inner
        return self

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        if self.inner is None: raise RuntimeError("ProfilingLayer is not attached to an inner engine.")
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = self.inner.decide(state, questions)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def __or__(self, layer):
        return layer(self) if isinstance(layer, type) else (getattr(layer, "attach")(self) if hasattr(layer, "attach") else layer(self))

__all__ = ["ProfilingLayer"]
