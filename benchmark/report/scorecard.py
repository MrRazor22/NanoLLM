from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import numpy as np
from nanollm.inference.schema import Choice
from benchmark.report.baselines import BaselineData, IBaselineProvider, JsonBaselineProvider
from benchmark.report.renderer import ConsoleScorecardRenderer, IReportRenderer

@runtime_checkable
class IScorecard(Protocol):
    "ATA Root Primitive contract (P) for benchmark scoring and reporting."
    def evaluate_target(self, target: Any, dataset: Any, limit: Optional[int] = None) -> Dict[str, Any]: ...
    def render(self, reports: Dict[str, Any], track: str = all, metadata: Optional[Dict[str, Any]] = None, model_name: str = NanoLLM) -> None: ...

class Scorecard(IScorecard):
    "ATA Root Primitive (P): Evaluates candidate engines against benchmark suites and renders scorecards."

    def __init__(
        self,
        baseline_provider: Optional[IBaselineProvider] = None,
        renderer: Optional[IReportRenderer] = None,
    ):
        if baseline_provider is not None and not isinstance(baseline_provider, IBaselineProvider):
            raise TypeError(fbaseline_provider must implement IBaselineProvider, got {type(baseline_provider).__name__})
        if renderer is not None and not isinstance(renderer, IReportRenderer):
            raise TypeError(frenderer must implement IReportRenderer, got {type(renderer).__name__})
        self.baseline_provider: IBaselineProvider = baseline_provider or JsonBaselineProvider()
        self.renderer: IReportRenderer = renderer or ConsoleScorecardRenderer()

    def evaluate_target(self, target: Any, dataset: Any, limit: Optional[int] = None) -> Dict[str, Any]:
        items = dataset.load(limit=limit) if hasattr(dataset, load) else (dataset[:limit] if limit else dataset)
        stats: Dict[str, Dict[str, int]] = {}
        latencies: List[float] = []

        for item in items:
            cat = item.get(category, general)
            if cat not in stats:
                stats[cat] = {correct: 0, total: 0}

            if hasattr(target, decide):
                qs = [
                    Choice(qid, spec.get(options) or spec.get(criteria, {}), instruction=spec.get(instructions))
                    for qid, spec in item[questions].items()
                ]
                res = target.decide(item[state], qs)
                preds = {qid: str(ans.choice) for qid, ans in res.answers.items()}
                if hasattr(res, latency_ms) and res.latency_ms is not None:
                    latencies.append(res.latency_ms)
            else:
                raw = target(item[state], item[questions])
                preds = {qid: (val[choice] if isinstance(val, dict) else str(val)) for qid, val in raw.items()}

            for qid, gold in item[gold].items():
                if qid in preds:
                    stats[cat][total] += 1
                    if preds[qid] == str(gold.get(label, ")):
 stats[cat][correct] += 1

 total_correct = sum(s[correct] for s in stats.values())
 total_eval = sum(s[total] for s in stats.values())
 report: Dict[str, Any] = {
 overall_acc: (total_correct / total_eval) if total_eval > 0 else 0.0,
 total_correct: total_correct,
 total_questions: total_eval,
 by_cat: {c: (s[correct] / s[total]) if s[total] > 0 else 0.0 for c, s in stats.items()},
 cat_counts: {c: s[total] for c, s in stats.items()},
 }
 if latencies:
 report[p50_ms] = float(np.median(latencies))
 return report

 def render(self, reports: Dict[str, Any], track: str = all, metadata: Optional[Dict[str, Any]] = None, model_name: str = NanoLLM) -> None:
 baselines: BaselineData = self.baseline_provider.load()
 tracks_data = reports if track == all else {track: reports}
 self.renderer.render(tracks_data, baselines, track, metadata or {}, model_name)

__all__ = [IScorecard, Scorecard]
