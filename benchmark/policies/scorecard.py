import orjson
from pathlib import Path
from typing import Mapping, Optional, Protocol, Union
from benchmark.policies.scorer import BenchmarkReport

DEFAULT_SCORECARD_FILE = Path(__file__).resolve().parent.parent / "scorecard.json"

SuiteEntry = Mapping[str, Union[float, int, str]]
SuiteScores = Mapping[str, SuiteEntry]
ScorecardScores = Mapping[str, SuiteScores]

class IScorecardPolicy(Protocol):
    def save(self, report: BenchmarkReport) -> None: ...
    def load(self, suite: Optional[str] = None) -> Mapping[str, dict]: ...

class DiskScorecardPolicy(IScorecardPolicy):
    def __init__(self, scorecard_path: Optional[Union[str, Path]] = None):
        self.scorecard_path = Path(scorecard_path or DEFAULT_SCORECARD_FILE)

    def save(self, report: BenchmarkReport) -> None:
        data = self.load()
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

        self.scorecard_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def load(self, suite: Optional[str] = None) -> Mapping[str, dict]:
        data = orjson.loads(self.scorecard_path.read_bytes()) if self.scorecard_path.exists() and self.scorecard_path.stat().st_size > 0 else {}
        return data.get(suite, {}) if suite is not None else data
