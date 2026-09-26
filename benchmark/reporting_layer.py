from typing import Any, Dict, Optional
from benchmark.evaluator import IEvaluator

class ReportingEvaluatorLayer:
    """ATA Composable Layer: transparently decorates IEvaluator to log model accuracy and latency scorecards."""

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

    @classmethod
    def render_scorecard(cls, report: Dict[str, Any], track: str = "all", model_name: str = "Model") -> None:
        tot_q = report.get("total_questions", 0)
        overall_acc = report.get("overall_acc", 0.0) * 100.0
        by_cat = report.get("by_cat", {})
        cat_counts = report.get("cat_counts", {})

        sep, dash = "=" * 65, "-" * 65
        print(f"\n{sep}\n{f'{model_name.upper()} SCORECARD ({track.upper()})':^65}\n{sep}")
        print(f"{'Category / Slice':35s} | {'Count':>8s} | {'Accuracy':>12s}\n{dash}")

        for cat, acc in sorted(by_cat.items()):
            c_title = cat.replace("_", " ").title()
            c_cnt = str(cat_counts.get(cat, "-"))
            print(f"  - {c_title:31s} | {c_cnt:>8s} | {acc * 100.0:11.1f}%")

        print(f"{dash}\n{'OVERALL ACCURACY':35s} | {str(tot_q):>8s} | {overall_acc:11.1f}%")
        if "p50_ms" in report:
            print(f"{'P50 INFERENCE LATENCY':35s} | {'-':>8s} | {report['p50_ms']:11.1f} ms")
        print(f"{sep}\n")

    def evaluate(self, source: Any, limit: Optional[int] = None) -> Dict[str, Any]:
        if self.inner is None:
            raise RuntimeError("ReportingEvaluatorLayer is not attached to an inner evaluator.")
        report = self.inner.evaluate(source, limit=limit)
        model_name = getattr(self.inner, "name", "NanoLLM")
        self.render_scorecard(report, track=self.track, model_name=model_name)
        return report

__all__ = ["ReportingEvaluatorLayer"]
