import time
from typing import Optional, Sequence
from inference.questions import DecisionResult, DecisionState, Question
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer

def synchronize_device() -> None:
    """Domain Transform: Synchronizes CUDA compute target if active."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, AttributeError):
        pass

class ProfilingLayer(IDecisionEngine):
    """Layer Decorator (lambda: P -> P): Measures wall-clock decision latency."""
    def __init__(self, inner: Optional[IDecisionEngine] = None):
        self.inner = inner

    def __call__(self, inner: IDecisionEngine) -> "ProfilingLayer":
        self.inner = inner
        return self

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        if self.inner is None: raise RuntimeError("ProfilingLayer is not attached to an inner engine.")
        synchronize_device()
        t0 = time.perf_counter()
        result = self.inner.decide(state, questions)
        synchronize_device()
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
