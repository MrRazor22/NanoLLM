import json
from pathlib import Path
from typing import Dict, Optional, Protocol, Union, runtime_checkable
from benchmark.policies.scorer import BenchmarkReport

DEFAULT_SCORES_FILE = Path(__file__).resolve().parent.parent / "scores.json"

@runtime_checkable
class ILedgerPolicy(Protocol):
    """Sovereign opinion sub-primitive: manages storage, retrieval, and caching of benchmark reports."""
    def get(self, model: str, suite: str = "default") -> Optional[Dict[str, float]]: ...
    def put(self, report: BenchmarkReport) -> None: ...
    def all(self, suite: Optional[str] = None) -> Dict[str, dict]: ...

class DiskLedgerPolicy(ILedgerPolicy):
    """Local JSON filesystem ledger policy with per-suite caching."""

    def __init__(self, scores_path: Optional[Union[str, Path]] = None):
        self.scores_path = Path(scores_path or DEFAULT_SCORES_FILE)

    def get(self, model: str, suite: str = "default") -> Optional[Dict[str, float]]:
        data = json.loads(self.scores_path.read_text(encoding="utf-8")) if self.scores_path.exists() else {}
        suite_data = data.get(suite, {})
        scores = {t: entry[model] for t, entry in suite_data.items() if isinstance(entry, dict) and model in entry}
        return scores if scores else None

    def put(self, report: BenchmarkReport) -> None:
        data = json.loads(self.scores_path.read_text(encoding="utf-8")) if self.scores_path.exists() else {}
        suite_data = data.setdefault(report.suite or "default", {})
        suite_data["count"] = report.count

        for cat, acc in report.slices.items():
            entry = suite_data.setdefault(cat, {})
            entry["count"], entry[report.model] = report.counts.get(cat, 0), round(acc, 4)
        overall = suite_data.setdefault("OVERALL", {})
        overall["count"], overall[report.model] = report.count, round(report.overall, 4)
        lat = suite_data.setdefault("LATENCY_P50_MS", {})
        lat["count"], lat[report.model] = "-", round(report.p50_ms, 2)
        if report.brier_score > 0:
            brier_entry = suite_data.setdefault("BRIER_SCORE", {})
            brier_entry["count"], brier_entry[report.model] = "-", round(report.brier_score, 4)

        self.scores_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def all(self, suite: Optional[str] = None) -> Dict[str, dict]:
        data = json.loads(self.scores_path.read_text(encoding="utf-8")) if self.scores_path.exists() else {}
        return data.get(suite, {}) if suite is not None else data
