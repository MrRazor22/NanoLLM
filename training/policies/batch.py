import itertools
import random
from typing import Iterator, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase

@runtime_checkable
class IBatch(Protocol):
    """Policy contract for partitioning and ordering domain cases into batches."""
    def batch(self, cases: Sequence[DecisionCase], batch_size: int = 16) -> Iterator[Sequence[DecisionCase]]: ...

def estimate_case_length(case: DecisionCase) -> int:
    """Pure Transform: Estimates token footprint of a case for bucketed batching."""
    state_len = len(case.state) // 4 if isinstance(case.state, str) else 30
    opt_len = sum(len(c) // 4 for c in case.candidates) if case.candidates else 10
    return state_len + opt_len

class BucketedBatch(IBatch):
    """Batches cases sorted by length heuristic to minimize sequence padding waste."""

    def __init__(self, shuffle: bool = True):
        self.shuffle = shuffle

    def batch(self, cases: Sequence[DecisionCase], batch_size: int = 16) -> Iterator[Sequence[DecisionCase]]:
        if not cases:
            return
        sorted_cases = sorted(cases, key=estimate_case_length)
        chunks = list(itertools.batched(sorted_cases, batch_size))
        if self.shuffle:
            random.shuffle(chunks)
        yield from chunks

class SequentialBatch(IBatch):
    """Deterministic linear chunking without reordering, ideal for evaluation."""

    def batch(self, cases: Sequence[DecisionCase], batch_size: int = 16) -> Iterator[Sequence[DecisionCase]]:
        yield from itertools.batched(cases, batch_size)
