import json
from pathlib import Path
from typing import Dict, Optional, Protocol, Union, runtime_checkable
from benchmark.scorecard import BenchmarkReport

DEFAULT_SCORES_FILE = Path(__file__).resolve().parent / "scores.json"

@runtime_checkable
class ILedgerPolicy(Protocol):
    """Sovereign opinion sub-primitive: manages storage, retrieval, and signature validation of benchmark reports."""
    def get(self, model: str, suite: str = "default", signature: Optional[str] = None) -> Optional[Dict[str, float]]: ...
    def put(self, report: BenchmarkReport, signature: Optional[str] = None) -> None: ...
    def all(self, suite: Optional[str] = None) -> Dict[str, dict]: ...

class DiskLedgerPolicy(ILedgerPolicy):
    """Local JSON filesystem ledger policy with per-suite caching and signature validation."""

    def __init__(self, scores_path: Optional[Union[str, Path]] = None):
        self.scores_path = Path(scores_path or DEFAULT_SCORES_FILE)

    def _read(self) -> Dict[str, dict]:
        if not self.scores_path.exists():
            return {}
        try:
            return json.loads(self.scores_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get(self, model: str, suite: str = "default", signature: Optional[str] = None) -> Optional[Dict[str, float]]:
        ledger = self._read()
        suite_data = ledger.get(suite)
        if not suite_data:
            return None
        if signature is not None and suite_data.get("signature") != signature:
            return None
        scores = {}
        for track, entry in suite_data.items():
            if isinstance(entry, dict) and model in entry:
                scores[track] = entry[model]
        return scores if scores else None

    def put(self, report: BenchmarkReport, signature: Optional[str] = None) -> None:
        ledger = self._read()
        suite = report.suite or "default"
        suite_data = ledger.setdefault(suite, {})
        if signature is not None:
            if suite_data.get("signature") != signature:
                suite_data.clear()
                suite_data["signature"] = signature
            suite_data["count"] = report.count

        for cat, acc in report.slices.items():
            entry = suite_data.setdefault(cat, {})
            entry["count"], entry[report.model] = report.counts.get(cat, 0), round(acc, 4)
        overall = suite_data.setdefault("OVERALL", {})
        overall["count"], overall[report.model] = report.count, round(report.overall, 4)
        lat = suite_data.setdefault("LATENCY_P50_MS", {})
        lat["count"], lat[report.model] = "-", round(report.p50_ms, 2)

        self.scores_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    def all(self, suite: Optional[str] = None) -> Dict[str, dict]:
        ledger = self._read()
        return ledger.get(suite, {}) if suite is not None else ledger
