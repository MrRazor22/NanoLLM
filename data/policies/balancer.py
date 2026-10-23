from collections import Counter, defaultdict
import logging
import random
from typing import Optional, Protocol, Sequence, runtime_checkable
import xxhash

from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class ICaseBalancer(Protocol):
    def bucket(
        self,
        case: DecisionCase,
        instruction_counts: Optional[Counter[tuple[str, str]]] = None,
        choice_counts: Optional[Counter[tuple[str, str]]] = None,
    ) -> tuple: ...
    def balance(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]: ...

class RoundRobinCaseBalancer(ICaseBalancer):
    def __init__(self, repetition_threshold: int = 2, seed: int = 42):
        self.repetition_threshold = repetition_threshold
        self.seed = seed

    def bucket(
        self,
        case: DecisionCase,
        instruction_counts: Optional[Counter[tuple[str, str]]] = None,
        choice_counts: Optional[Counter[tuple[str, str]]] = None,
    ) -> tuple:
        source = case.meta.source
        candidate_type = case.meta.candidate_type
        choice = case.choice()

        if instruction_counts is not None:
            is_repeated_inst = instruction_counts.get((source, case.instruction), 0) >= self.repetition_threshold
            instruction_signature = xxhash.xxh64_intdigest(case.instruction.encode()) if is_repeated_inst else 0
        else:
            instruction_signature = xxhash.xxh64_intdigest(case.instruction.encode()) if case.instruction else 0

        if choice_counts is not None:
            choice_signature = choice if choice_counts.get((source, choice), 0) >= self.repetition_threshold else ""
        else:
            choice_signature = choice

        return (
            source,
            candidate_type,
            instruction_signature,
            len(case.candidates),
            case.target_index(),
            choice_signature,
        )

    def balance(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]:
        if not cases:
            return ()

        inst_counts = Counter((c.meta.source, c.instruction) for c in cases if c.instruction)
        choice_counts = Counter((c.meta.source, c.choice()) for c in cases if c.choice())

        rng = random.Random(self.seed)
        buckets = defaultdict(list)
        for c in cases:
            buckets[self.bucket(c, inst_counts, choice_counts)].append(c)

        groups = list(buckets.values())
        for group in groups:
            rng.shuffle(group)

        selected: list[DecisionCase] = []
        exhausted_count = 0
        num_groups = len(groups)

        while exhausted_count < num_groups:
            exhausted_count = 0
            for group in groups:
                if group:
                    selected.append(group.pop())
                else:
                    exhausted_count += 1

        return tuple(selected)
