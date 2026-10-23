from collections import Counter
from dataclasses import dataclass
import logging
from typing import Optional, Protocol, Sequence, runtime_checkable
import xxhash

from data.case import DecisionCase

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class CategoryBucket:
    source: str
    candidate_type: str
    instruction_signature: int
    num_candidates: int
    target_index: int
    choice_signature: str

@runtime_checkable
class IBucketPolicy(Protocol):
    def prepare(self, cases: Sequence[DecisionCase]) -> None: ...
    def bucket(self, case: DecisionCase) -> CategoryBucket: ...

class CaseBucketPolicy(IBucketPolicy):
    def __init__(self, repetition_threshold: int = 2):
        self.repetition_threshold = repetition_threshold
        self.instruction_counts: Counter[tuple[str, str]] = Counter()
        self.choice_counts: Counter[tuple[str, str]] = Counter()

    def prepare(self, cases: Sequence[DecisionCase]) -> None:
        self.instruction_counts = Counter((c.meta.source, c.instruction) for c in cases if c.instruction)
        self.choice_counts = Counter((c.meta.source, c.choice()) for c in cases if c.choice())

    def bucket(self, case: DecisionCase) -> CategoryBucket:
        source = case.meta.source
        candidate_type = case.meta.candidate_type
        choice = case.choice()

        is_repeated_inst = self.instruction_counts.get((source, case.instruction), 0) >= self.repetition_threshold
        instruction_signature = xxhash.xxh64_intdigest(case.instruction.encode()) if is_repeated_inst else 0

        is_repeated_choice = self.choice_counts.get((source, choice), 0) >= self.repetition_threshold
        choice_signature = choice if is_repeated_choice else ""

        return CategoryBucket(
            source=source,
            candidate_type=candidate_type,
            instruction_signature=instruction_signature,
            num_candidates=len(case.candidates),
            target_index=case.target_index(),
            choice_signature=choice_signature,
        )

__all__ = ["CategoryBucket", "IBucketPolicy", "CaseBucketPolicy"]
