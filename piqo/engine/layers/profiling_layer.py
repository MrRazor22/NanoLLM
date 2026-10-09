import time
from typing import TYPE_CHECKING, Optional, Sequence
import torch
from piqo.engine.questions import Composable, DecisionResult, DecisionState, Question

if TYPE_CHECKING:
    from piqo.engine.decision_engine import IDecisionEngine

class ProfilingLayer(Composable):
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

__all__ = ["ProfilingLayer"]
