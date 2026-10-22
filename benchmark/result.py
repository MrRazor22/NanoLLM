from dataclasses import dataclass
from benchmark.policies.scorer import DecisionMetrics

@dataclass(frozen=True)
class BenchmarkResult:
    suite: str
    engine: str
    metrics: DecisionMetrics
