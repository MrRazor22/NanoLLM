from typing import Protocol, Sequence
from benchmark.suites.case import BenchmarkCase

class IBenchmarkSuite(Protocol):
    def test_cases(self) -> Sequence[BenchmarkCase]: ...
