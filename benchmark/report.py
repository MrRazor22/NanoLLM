from dataclasses import dataclass, field
from typing import Mapping

@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Pure domain DTO: verified evaluation report for a decision engine on a benchmark suite."""
    model: str
    overall: float
    count: int
    p50_ms: float = 0.0
    slices: Mapping[str, float] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    suite: str = "default"
