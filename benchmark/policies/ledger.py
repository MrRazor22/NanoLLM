import orjson
from pathlib import Path
from typing import Mapping, Optional, Protocol, Union, runtime_checkable
from benchmark.policies.scorer import BenchmarkReport

DEFAULT_SCORES_FILE = Path(__file__).resolve().parent.parent / "scores.json"

SuiteEntry = Mapping[str, Union[float, int, str]]
SuiteScores = Mapping[str, SuiteEntry]
LedgerScores = Mapping[str, SuiteScores]

class ILedgerPolicy(Protocol):
    def lookup(self, model: str, suite: str = "default", expected_count: Optional[int] = None) -> Optional[Mapping[str, float]]: ...
    def record(self, report: BenchmarkReport) -> None: ...
    def entries(self, suite: Optional[str] = None) -> Mapping[str, dict]: ...

class DiskLedgerPolicy(ILedgerPolicy):
    def __init__(self, scores_path: Optional[Union[str, Path]] = None):
        self.scores_path = Path(scores_path or DEFAULT_SCORES_FILE)

    def lookup(self, model: str, suite: str = "default", expected_count: Optional[int] = None) -> Optional[Mapping[str, float]]:
        suite_data = self.entries(suite)
        if expected_count is not None and suite_data.get("count") != expected_count:
            return None
        scores = {t: entry[model] for t, entry in suite_data.items() if isinstance(entry, dict) and model in entry}
        return scores if scores else None

    def record(self, report: BenchmarkReport) -> None:
        data = self.entries()
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
        if report.regression_mse > 0:
            reg_entry = suite_data.setdefault("REGRESSION_MSE", {})
            reg_entry["count"], reg_entry[report.model] = "-", round(report.regression_mse, 4)

        self.scores_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def entries(self, suite: Optional[str] = None) -> Mapping[str, dict]:
        data = orjson.loads(self.scores_path.read_bytes()) if self.scores_path.exists() and self.scores_path.stat().st_size > 0 else {}
        return data.get(suite, {}) if suite is not None else data
