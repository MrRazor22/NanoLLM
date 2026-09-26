from typing import Optional, Sequence
import time
import torch
from pipeline import PipelineComposable
from nanollm.inference.decision_engine import IDecisionEngine
from nanollm.inference.schema import DecisionResult, Question

class DecisionEngineLayer(PipelineComposable, IDecisionEngine):
    """ATA Composable Layer Contract (λ: P -> P) for DecisionEngine."""
    def __init__(self, inner: Optional[IDecisionEngine] = None):
        if inner is not None and not isinstance(inner, IDecisionEngine):
            raise TypeError(f"inner must implement IDecisionEngine, got {type(inner).__name__}")
        self.inner = inner

    def attach(self, inner: IDecisionEngine) -> "DecisionEngineLayer":
        if not isinstance(inner, IDecisionEngine):
            raise TypeError(f"inner must implement IDecisionEngine, got {type(inner).__name__}")
        self.inner = inner
        return self

    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult:
        if self.inner is None:
            raise RuntimeError(f"{self.__class__.__name__} is not attached to an inner engine.")
        return self.inner.decide(state, questions)

class ProfilingLayer(DecisionEngineLayer):
    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult:
        if self.inner is None:
            raise RuntimeError("ProfilingLayer is not attached to an inner engine.")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = self.inner.decide(state, questions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result.latency_ms = (time.perf_counter() - start) * 1000.0
        return result
