import time
from typing import Optional, Sequence
from inference.questions import DecisionResult, DecisionState, Question
from inference.decision_engine import IDecisionEngine

def _sync_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass

class ProfilingLayer(IDecisionEngine):
    """Layer Decorator (lambda: P -> P): Measures wall-clock decision latency."""
    def __init__(self, inner: Optional[IDecisionEngine] = None):
        self.inner = inner

    def __call__(self, inner: IDecisionEngine) -> "ProfilingLayer":
        self.inner = inner
        return self

    attach = __call__

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        if self.inner is None: raise RuntimeError("ProfilingLayer is not attached to an inner engine.")
        _sync_cuda()
        t0 = time.perf_counter()
        result = self.inner.decide(state, questions)
        _sync_cuda()
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def __or__(self, layer) -> IDecisionEngine:
        return layer(self)
