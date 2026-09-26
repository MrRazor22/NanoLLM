from typing import Any, Dict, Mapping, Optional, Sequence, Union
from benchmark.competitor import CompetitorScorecard
from benchmark.dataset import BenchmarkDataset
from benchmark.runner import IBenchmarkRunner

class ScorecardLayer(IBenchmarkRunner):
    """ATA Composable Layer: intercepts benchmark evaluation, streams suite banners, and renders comparative scorecards."""

    def __init__(
        self,
        inner: Optional[IBenchmarkRunner] = None,
        track: str = "all",
        compare: bool = False,
        model_name: str = "NanoLLM Champion",
        show_banners: bool = True,
    ):
        self.inner = inner
        self.track = track
        self.compare = compare
        self.model_name = model_name
        self.show_banners = show_banners
        self.scorecard = CompetitorScorecard()

    def attach(self, inner: IBenchmarkRunner) -> "ScorecardLayer":
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

    def evaluate(
        self,
        sources: Union[Mapping[str, Any], Sequence[Any], Any],
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.inner is None:
            raise RuntimeError("ScorecardLayer is not attached to an inner runner.")

        # Single source (e.g. unit test or single dataset instance)
        if not isinstance(sources, Mapping):
            report = self.inner.evaluate(sources, limit=limit)
            if self.compare:
                self.scorecard.render({self.track: report}, track=self.track, model_name=self.model_name)
            return report

        reports: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {}

        for key, src in sources.items():
            ds = BenchmarkDataset(src()) if isinstance(src, type) else (BenchmarkDataset(src) if hasattr(src, "extract") else src)
            meta = {
                "display_name": getattr(ds, "display_name", key.replace("_", " ").title()),
                "category_labels": getattr(ds, "category_labels", {}),
            }
            if self.show_banners and len(sources) > 1:
                title = meta["display_name"].upper()
                print(f"\n{'='*70}\n>>> Running Benchmark Suite: {title}\n{'='*70}", flush=True)

            reports[key] = self.inner.evaluate(ds, limit=limit)
            metadata[key] = meta

        # Render unified competitive matrix at the conclusion
        self.scorecard.render(reports, track=self.track, metadata=metadata, model_name=self.model_name)
        return reports

__all__ = ["ScorecardLayer"]
