from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union
import torch
from nanollm.engine.questions import DecisionResult, DecisionState, Question

if TYPE_CHECKING:
    from nanollm.engine.decision_engine import IDecisionEngine

class ProfilingLayer:
    """Layer Decorator (lambda: P -> P): Measures wall-clock decision latency."""

    def __init__(self, inner: Optional["IDecisionEngine"] = None):
        self.inner = inner

    def attach(self, inner: "IDecisionEngine") -> "ProfilingLayer":
        self.inner = inner
        return self

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        if self.inner is None:
            raise RuntimeError("ProfilingLayer is not attached to an inner engine.")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = self.inner.decide(state, questions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def decide_batch(
        self, batch: Sequence[Tuple[DecisionState, Sequence[Question]]]
    ) -> Sequence[DecisionResult]:
        return [self.decide(s, q) for s, q in batch]

    def add(self, layer: Union[type, "ProfilingLayer"]) -> "IDecisionEngine":
        if isinstance(layer, type):
            return layer(self)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer

    def __or__(self, layer: Union[type, "ProfilingLayer"]) -> "IDecisionEngine":
        return self.add(layer)

import time

__all__ = ["ProfilingLayer"]
