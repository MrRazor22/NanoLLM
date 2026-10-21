import time
from typing import Sequence

from data.case import DecisionCase
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.options import IOption
from model.device import synchronize_device

class ProfilingLayer(IDecisionEngine):
    """Layer Decorator (lambda: P -> P): Transparent hardware synchronization and latency profiling."""

    def __init__(self, inner: IDecisionEngine | None = None):
        self.inner = inner
        self.last_latency_ms: float = 0.0

    def __call__(self, inner: IDecisionEngine) -> "ProfilingLayer":
        self.inner = inner
        return self

    def predict_candidates(self, state: str, candidates: Sequence[str]) -> Sequence[float]:
        if not self.inner:
            raise RuntimeError("ProfilingLayer has no inner engine.")
        synchronize_device()
        t0 = time.perf_counter()
        res = self.inner.predict_candidates(state, candidates)
        synchronize_device()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return res

    def decide(self, state: str, option: IOption) -> IOption:
        if not self.inner:
            raise RuntimeError("ProfilingLayer has no inner engine.")
        synchronize_device()
        t0 = time.perf_counter()
        res = self.inner.decide(state, option)
        synchronize_device()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return res

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
