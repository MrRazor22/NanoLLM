from typing import Mapping, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase

@runtime_checkable
class IBenchmarkSuite(Protocol):
    """Sovereign benchmark suite contract: provides evaluation cases, identity, and published baselines."""
    @property
    def name(self) -> str: ...

    @property
    def published_baselines(self) -> Mapping[str, float]: ...

    def load_cases(self) -> Sequence[DecisionCase]: ...
