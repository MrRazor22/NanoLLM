from typing import Optional, Protocol, runtime_checkable
import random
from data.case import DecisionCase

@runtime_checkable
class ICaseTransform(Protocol):
    def transform(self, case: DecisionCase) -> DecisionCase: ...

class ShuffleCandidates(ICaseTransform):
    def __init__(self, seed: int = 42):
        self.random_generator = random.Random(seed)

    def transform(self, case: DecisionCase) -> DecisionCase:
        if len(case.candidates) <= 1:
            return case
        permuted = list(case.candidates)
        self.random_generator.shuffle(permuted)
        return DecisionCase(
            state=case.state,
            candidates=tuple(permuted),
            instruction=case.instruction,
            probabilities=case.probabilities,
            meta=case.meta,
        )

class IdentityTransform(ICaseTransform):
    def transform(self, case: DecisionCase) -> DecisionCase:
        return case
