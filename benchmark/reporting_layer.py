from typing import Any, Dict, List, Optional
from benchmark.evaluator import IEvaluator
from benchmark.reporter import print_scorecard

class ReportingEvaluatorLayer:
    def __init__(self, inner: Optional[IEvaluator] = None, track: str = "all"):
        self.inner = inner
        self.track = track

    def attach(self, inner: IEvaluator) -> "ReportingEvaluatorLayer":
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
            raise RuntimeError("ReportingEvaluatorLayer is not attached to an inner evaluator.")
        report = self.inner.evaluate(items)
        print_scorecard(report, track=self.track)
        return report

__all__ = ["ReportingEvaluatorLayer"]
