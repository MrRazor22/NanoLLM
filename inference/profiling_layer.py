import logging
import time
from typing import Self

from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.options import IAnswer, IOption
from model.device import synchronize_device

logger = logging.getLogger(__name__)

class ProfilingLayer(IDecisionEngine):
    def __init__(self, inner: IDecisionEngine | None = None):
        self.inner = inner
        self.last_latency_ms: float = 0.0

    def __call__(self, inner: IDecisionEngine) -> Self:
        self.inner = inner
        return self

    def decide(self, state: str, option: IOption) -> IAnswer:
        if not self.inner:
            raise RuntimeError("ProfilingLayer has no inner engine.")
        synchronize_device()
        t0 = time.perf_counter()
        res = self.inner.decide(state, option)
        synchronize_device()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("Latency measured at %.2f ms", self.last_latency_ms)
        return res

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)
