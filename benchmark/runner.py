from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Union
import numpy as np
import torch

from nanollm.inference.engine import DEFAULT_CHECKPOINT, DecisionEngine
from nanollm.inference.profiling_layer import ProfilingLayer
from nanollm.inference.schema import Choice
from benchmark.dataset import BenchmarkDataset

class IBenchmarkRunner(Protocol):
    """ATA Root Contract: Universal benchmark evaluation interface."""
    def evaluate(
        self, sources: Union[Mapping[str, Any], Sequence[Any], Any], limit: Optional[int] = None
    ) -> Dict[str, Any]: ...

class BenchmarkRunner(IBenchmarkRunner):
    """Benchmark Primitive: Executes candidate engine against benchmark sources and computes metrics."""

    def __init__(self, target: Any):
        self.target = target

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        profile: bool = True,
    ) -> "BenchmarkRunner":
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        engine = DecisionEngine.from_checkpoint(checkpoint_path, device=dev)
        if profile:
            engine = engine | ProfilingLayer()
        return cls(engine)

    def add(self, layer: Any, **kwargs: Any) -> Any:
        if isinstance(layer, type):
            return layer(self, **kwargs)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer(self, **kwargs)

    def __or__(self, layer: Any) -> Any:
        return self.add(layer)

    def _predict(self, state: str, questions: Dict[str, Any]) -> tuple[Dict[str, str], Optional[float]]:
        if hasattr(self.target, "decide"):
            qs = [
                Choice(qid, spec.get("options") or spec.get("criteria", {}), instruction=spec.get("instructions"))
                for qid, spec in questions.items()
            ]
            res = self.target.decide(state, qs)
            preds = {qid: str(ans.choice) for qid, ans in res.answers.items()}
            return preds, getattr(res, "latency_ms", None)

        raw = self.target(state, questions)
        preds = {qid: (val["choice"] if isinstance(val, dict) else str(val)) for qid, val in raw.items()}
        return preds, None

    def _evaluate_dataset(self, source: Any, limit: Optional[int] = None) -> Dict[str, Any]:
        items = source.load(limit=limit) if hasattr(source, "load") else (source[:limit] if limit else source)
        stats: Dict[str, Dict[str, int]] = {}
        latencies: List[float] = []

        for item in items:
            cat = item.get("category", "general")
            if cat not in stats:
                stats[cat] = {"correct": 0, "total": 0}

            preds, lat = self._predict(item["state"], item["questions"])
            if lat is not None:
                latencies.append(lat)

            for qid, gold in item["gold"].items():
                if qid in preds:
                    stats[cat]["total"] += 1
                    if preds[qid] == str(gold.get("label", "")):
                        stats[cat]["correct"] += 1

        total_correct = sum(s["correct"] for s in stats.values())
        total_eval = sum(s["total"] for s in stats.values())
        report: Dict[str, Any] = {
            "overall_acc": (total_correct / total_eval) if total_eval > 0 else 0.0,
            "total_correct": total_correct,
            "total_questions": total_eval,
            "by_cat": {c: (s["correct"] / s["total"]) if s["total"] > 0 else 0.0 for c, s in stats.items()},
            "cat_counts": {c: s["total"] for c, s in stats.items()},
        }
        if latencies:
            report["p50_ms"] = float(np.median(latencies))
        return report

    def evaluate(
        self,
        sources: Union[Mapping[str, Any], Sequence[Any], Any],
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        if isinstance(sources, Mapping):
            reports: Dict[str, Any] = {}
            for key, src in sources.items():
                ds = BenchmarkDataset(src()) if isinstance(src, type) else (BenchmarkDataset(src) if hasattr(src, "extract") else src)
                reports[key] = self._evaluate_dataset(ds, limit=limit)
            return reports

        return self._evaluate_dataset(sources, limit=limit)

__all__ = ["IBenchmarkRunner", "BenchmarkRunner"]
