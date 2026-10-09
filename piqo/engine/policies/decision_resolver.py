import math
from typing import Dict, Mapping, Protocol, Sequence, runtime_checkable
from piqo.engine.questions import (
    Answer,
    Choice,
    ChoiceResult,
    Noul,
    NoulResult,
    Question,
    Score,
    ScoreResult,
)

@runtime_checkable
class IResolver(Protocol):
    """Injected Opinion Sub-Primitive: Decodes raw float scores into typed domain Answers."""

    def resolve(
        self,
        questions: Sequence[Question],
        slots: Sequence[Sequence[int]],
        scores: Sequence[float],
    ) -> Mapping[str, Answer]:
        ...

class DecisionResolver(IResolver):
    """Pure mathematical resolver: decodes raw float scores with zero framework dependencies."""

    def resolve(
        self,
        questions: Sequence[Question],
        slots: Sequence[Sequence[int]],
        scores: Sequence[float],
    ) -> Mapping[str, Answer]:
        answers: Dict[str, Answer] = {}
        for q, slot_indices in zip(questions, slots):
            if isinstance(q, Choice):
                opt_keys = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                raw = [scores[idx] for idx in slot_indices]
                max_l = max(raw)
                exp_v = [math.exp(x - max_l) for x in raw]
                sum_e = sum(exp_v)
                probs = [e / sum_e for e in exp_v]
                best_i = probs.index(max(probs))
                answers[q.name] = ChoiceResult(
                    choice=opt_keys[best_i],
                    confidence=probs[best_i],
                    probabilities={k: p for k, p in zip(opt_keys, probs)},
                )
            elif isinstance(q, Noul):
                prob = 1.0 / (1.0 + math.exp(-scores[slot_indices[0]]))
                answers[q.name] = NoulResult(is_positive=prob >= 0.5, probability=prob)
            elif isinstance(q, Score):
                norm = 1.0 / (1.0 + math.exp(-scores[slot_indices[0]]))
                val = q.min_value + norm * (q.max_value - q.min_value)
                answers[q.name] = ScoreResult(score=val, normalized=norm)
        return answers

__all__ = ["IResolver", "DecisionResolver"]
