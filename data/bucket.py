from dataclasses import dataclass
from typing import Sequence
import xxhash

from data.case import DecisionCase

@dataclass(frozen=True, slots=True)
class CategoryBucket:
    unique_keys: Sequence[tuple]
    repeated_instructions: frozenset[tuple[str, str]]
    repeated_options: frozenset[tuple[str, str]]

    def key(self, case: DecisionCase) -> tuple:
        source = case.meta.source
        candidate_type = case.meta.candidate_type
        option = case.option()

        instruction_signature = (
            xxhash.xxh64_intdigest(case.instruction.encode())
            if (source, case.instruction) in self.repeated_instructions
            else 0
        )
        option_signature = option if (source, option) in self.repeated_options else ""

        return (
            source,
            candidate_type,
            instruction_signature,
            len(case.candidates),
            case.target_index(),
            option_signature,
        )

__all__ = ["CategoryBucket"]
