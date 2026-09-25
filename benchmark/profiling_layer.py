import time
from typing import Any, Dict, List, Optional
import numpy as np
from benchmark.evaluator import IEvaluator

class ProfilingEvaluatorLayer:
    def __init__(self, inner: Optional[IEvaluator] = None, warmup_runs: int = 5):
        self.inner = inner
        self.warmup_runs = warmup_runs

    def attach(self, inner: IEvaluator) -> "ProfilingEvaluatorLayer":
        self.inner = inner
        return self

    def add(self, layer: Any, **kwargs: Any) -> Any:
        if isinstance(layer, type):
            return layer(self, **kwargs)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer(self, **kwargs)

    def __or__(self, layer: Any) -> Any:
        return self.add(layer)

    def evaluate(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.inner is None:
            raise RuntimeError("ProfilingEvaluatorLayer is not attached to an inner evaluator.")

        if items and self.warmup_runs > 0 and hasattr(self.inner, "decide_fn"):
            sample = items[0]
            for _ in range(self.warmup_runs):
                self.inner.decide_fn(sample["state"], sample["questions"])

        latencies = []
        if hasattr(self.inner, "decide_fn"):
            base_decide = self.inner.decide_fn
            def timed_decide(state, questions):
                t0 = time.perf_counter()
                out = base_decide(state, questions)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                return out
            self.inner.decide_fn = timed_decide

        report = self.inner.evaluate(items)
        if latencies:
            report["p50_ms"] = float(np.median(latencies))
            report["p90_ms"] = float(np.percentile(latencies, 90))
        return report
