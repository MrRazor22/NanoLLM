import itertools
import random
from typing import Iterator, Protocol, Sequence, runtime_checkable
from data.case import ChoiceAnswer, DecisionCase, NoulAnswer

def label_key(case: DecisionCase) -> str:
    ans = case.option.answer
    if ans is None:
        return ""
    if isinstance(ans, ChoiceAnswer):
        return str(ans.choice)
    if isinstance(ans, NoulAnswer):
        return str(ans.decision)
    return str(ans.score)

@runtime_checkable
class IStratificationPolicy(Protocol):
    """Sub-Primitive Contract: Balances cases evenly across label classes."""
    def stream(self, cases: Sequence[DecisionCase], rng: random.Random) -> Iterator[DecisionCase]: ...
    def balance(self, cases: Sequence[DecisionCase], quota: int, rng: random.Random) -> list[DecisionCase]: ...

class RoundRobinStratification(IStratificationPolicy):
    """Fair round-robin class stratification ensuring balanced label representations."""

    def stream(self, cases: Sequence[DecisionCase], rng: random.Random) -> Iterator[DecisionCase]:
        if not cases:
            return

        by_label: dict[str, list[DecisionCase]] = {}
        for c in cases:
            by_label.setdefault(label_key(c), []).append(c)

        for g in by_label.values():
            rng.shuffle(g)

        groups = [g for g in by_label.values() if g]
        while groups:
            for g in list(groups):
                if g:
                    yield g.pop()
                else:
                    groups.remove(g)

    def balance(self, cases: Sequence[DecisionCase], quota: int, rng: random.Random) -> list[DecisionCase]:
        if not cases or quota <= 0:
            return []
        return list(itertools.islice(self.stream(cases, rng), quota))

__all__ = ["IStratificationPolicy", "RoundRobinStratification"]
