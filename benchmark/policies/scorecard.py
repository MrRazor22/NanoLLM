from dataclasses import asdict
import logging
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union
import orjson

from benchmark.policies.scorer import DecisionMetrics
from benchmark.result import BenchmarkResult

logger = logging.getLogger(__name__)

DEFAULT_SCORECARD_FILE = Path(__file__).resolve().parents[1] / "scorecard.json"

class IScorecardPolicy(Protocol):
    def record(self, result: BenchmarkResult) -> None: ...
    def load(self, suites: Optional[Sequence[str]] = None) -> Sequence[BenchmarkResult]: ...

class ScorecardPolicy(IScorecardPolicy):
    def __init__(self, path: Optional[Union[str, Path]] = None):
        self.path = Path(path or DEFAULT_SCORECARD_FILE)

    def record(self, result: BenchmarkResult) -> None:
        data = self._read_json()
        data.setdefault(result.suite, {})[result.engine] = asdict(result.metrics)
        self.path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        logger.info("Recorded scorecard entry for '%s' [%s]", result.suite, result.engine)

    def load(self, suites: Optional[Sequence[str]] = None) -> Sequence[BenchmarkResult]:
        data = self._read_json()
        active_suites = set(suites) if suites else set(data.keys())
        results: list[BenchmarkResult] = []

        for s_name, entries in data.items():
            if s_name not in active_suites or not isinstance(entries, dict):
                continue
            for eng_name, m in entries.items():
                if not isinstance(m, dict):
                    continue
                metrics = DecisionMetrics(
                    accuracy=float(m.get("accuracy", 0.0)),
                    latency_p50_ms=float(m.get("latency_p50_ms", 0.0)),
                    brier_score=float(m.get("brier_score", 0.0)),
                    category_accuracy={str(k): float(v) for k, v in m.get("category_accuracy", {}).items()},
                    category_counts={str(k): int(v) for k, v in m.get("category_counts", {}).items()},
                    count=int(m.get("count", 0)),
                )
                results.append(BenchmarkResult(suite=s_name, engine=eng_name, metrics=metrics))
        return results

    def _read_json(self) -> dict:
        return orjson.loads(self.path.read_bytes()) if self.path.exists() and self.path.stat().st_size > 0 else {}
