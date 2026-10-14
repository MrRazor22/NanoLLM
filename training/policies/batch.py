import random
from typing import Callable, Iterator, Optional, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase

@runtime_checkable
class IBatch(Protocol):
    """Policy contract for partitioning and ordering domain cases into batches."""
    def batch(self, cases: Sequence[DecisionCase], batch_size: int = 16) -> Iterator[Sequence[DecisionCase]]: ...

class BucketedBatch(IBatch):
    """Batches cases sorted by length heuristic to minimize sequence padding waste."""

    def __init__(
        self,
        key: Optional[Callable[[DecisionCase], int]] = None,
        shuffle: bool = True,
    ):
        self.key = key or self._estimate_length
        self.shuffle = shuffle

    @staticmethod
    def _estimate_length(case: DecisionCase) -> int:
        s_len = len(case.state) // 4 if isinstance(case.state, str) else 30
        q_len = sum(len(q.options) * 5 if hasattr(q, "options") and q.options else 10 for q in case.questions)
        return s_len + q_len

    def batch(self, cases: Sequence[DecisionCase], batch_size: int = 16) -> Iterator[Sequence[DecisionCase]]:
        if not cases:
            return
        sorted_cases = sorted(cases, key=self.key)
        chunks = [sorted_cases[i : i + batch_size] for i in range(0, len(sorted_cases), batch_size)]
        if self.shuffle:
            random.shuffle(chunks)
        yield from chunks

class SequentialBatch(IBatch):
    """Deterministic linear chunking without reordering, ideal for evaluation."""

    def batch(self, cases: Sequence[DecisionCase], batch_size: int = 16) -> Iterator[Sequence[DecisionCase]]:
        for i in range(0, len(cases), batch_size):
            yield cases[i : i + batch_size]
