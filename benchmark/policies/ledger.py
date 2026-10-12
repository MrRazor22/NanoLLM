import json
from pathlib import Path
from typing import Dict, Optional, Protocol, Union, runtime_checkable
from benchmark.scorecard import BenchmarkReport

DEFAULT_SCORES_FILE = Path(__file__).resolve().parent.parent / "scores.json"

@runtime_checkable
class ILedgerPolicy(Protocol):
    """Sovereign opinion sub-primitive: manages storage, retrieval, and signature validation of benchmark reports."""
    def get(self, model: str, signature: Optional[str] = None) -> Optional[Dict[str, float]]: ...
    def put(self, report: BenchmarkReport, signature: Optional[str] = None) -> None: ...
    def all(self) -> Dict[str, dict]: ...

class DiskLedgerPolicy(ILedgerPolicy):
    """Local JSON filesystem ledger policy with automatic dataset signature validation."""

    def __init__(self, scores_path: Optional[Union[str, Path]] = None):
        self.scores_path = Path(scores_path or DEFAULT_SCORES_FILE)

    def _read(self) -> Dict[str, dict]:
        if not self.scores_path.exists():
            return {}
        try:
            return json.loads(self.scores_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get(self, model: str, signature: Optional[str] = None) -> Optional[Dict[str, float]]:
        ledger = self._read()
        if signature is not None:
            cached_sig = ledger.get("DATASET", {}).get("signature")
            if cached_sig != signature:
                return None
        scores = {}
        for track, entry in ledger.items():
            if track != "DATASET" and model in entry:
                scores[track] = entry[model]
        return scores if scores else None

    def put(self, report: BenchmarkReport, signature: Optional[str] = None) -> None:
        ledger = self._read()
        for cat, acc in report.slices.items():
            entry = ledger.setdefault(cat, {})
            entry["count"], entry[report.model] = report.counts.get(cat, 0), round(acc, 4)
        overall = ledger.setdefault("OVERALL", {})
        overall["count"], overall[report.model] = report.count, round(report.overall, 4)
        lat = ledger.setdefault("LATENCY_P50_MS", {})
        lat["count"], lat[report.model] = "-", round(report.p50_ms, 2)
        if signature is not None:
            ledger.setdefault("DATASET", {})["signature"] = signature
            ledger["DATASET"]["count"] = report.count
        self.scores_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    def all(self) -> Dict[str, dict]:
        return self._read()

__all__ = ["ILedgerPolicy", "DiskLedgerPolicy"]
