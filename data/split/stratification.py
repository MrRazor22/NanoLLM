import random
from typing import Protocol, Sequence, runtime_checkable
from data.case import ChoiceAnswer, DecisionCase, NoulAnswer

@runtime_checkable
class IStratificationPolicy(Protocol):
    """Sub-Primitive Contract: Balances cases evenly across label classes."""
    def balance(self, cases: Sequence[DecisionCase], quota: int, rng: random.Random) -> list[DecisionCase]: ...

class RoundRobinStratification(IStratificationPolicy):
    """Fair round-robin class stratification ensuring balanced label representations."""

    def balance(self, cases: Sequence[DecisionCase], quota: int, rng: random.Random) -> list[DecisionCase]:
        if not cases or quota <= 0:
            return []

        by_label: dict[str, list[DecisionCase]] = {}
        for c in cases:
            ans = c.option.answer
            tgt = str(ans.choice if isinstance(ans, ChoiceAnswer)
                      else (ans.decision if isinstance(ans, NoulAnswer)
                            else ans.score)) if ans is not None else ""
            by_label.setdefault(tgt, []).append(c)

        for g in by_label.values():
            rng.shuffle(g)

        selected: list[DecisionCase] = []
        groups = [g for g in by_label.values() if g]
        idx = 0
        while len(selected) < quota and any(groups):
            g = groups[idx % len(groups)]
            if g:
                selected.append(g.pop())
            else:
                groups.remove(g)
                continue
            idx += 1

        return selected

__all__ = ["IStratificationPolicy", "RoundRobinStratification"]
