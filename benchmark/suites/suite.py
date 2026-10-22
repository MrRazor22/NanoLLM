from typing import Protocol, Sequence
from data.case import DecisionCase

class IBenchmarkSuite(Protocol):
    def test_cases(self) -> Sequence[DecisionCase]: ...
