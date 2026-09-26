from typing import Any, Optional, Sequence
import time
import torch
from nanollm.inference.engine import IDecisionEngine
from nanollm.inference.schema import DecisionResult, Question

class DecisionEngineLayer(IDecisionEngine):
    def __init__(self, inner: Optional[IDecisionEngine] = None):
        self.inner = inner

    def attach(self, inner: IDecisionEngine) -> "DecisionEngineLayer":
        self.inner = inner
        return self

    def add(self, layer: Any, **kwargs: Any) -> IDecisionEngine:
        if isinstance(layer, type):
            return layer(self, **kwargs)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer(self, **kwargs)

    def __or__(self, layer: Any) -> IDecisionEngine:
        return self.add(layer)

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
