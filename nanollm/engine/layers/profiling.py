from typing import Sequence
import time
import torch
from nanollm.engine.engine import IDecisionEngine
from nanollm.schema import DecisionResult, Question

class DecisionEngineLayer(IDecisionEngine):
    def __init__(self, inner: IDecisionEngine):
        self.inner = inner

    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult:
        return self.inner.decide(state, questions)

class ProfilingLayer(DecisionEngineLayer):
    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = self.inner.decide(state, questions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result.latency_ms = (time.perf_counter() - start) * 1000.0
        return result
