from typing import Dict, Mapping, Protocol, Sequence, runtime_checkable
import scipy.special as sp
from inference.questions import (
    Answer, Choice, ChoiceResult, Noul, NoulResult, Question, Score, ScoreResult,
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
    """Numerically stable resolver: decodes raw float scores using industry-standard primitives."""

    def resolve(
        self,
        questions: Sequence[Question],
        slots: Sequence[Sequence[int]],
        scores: Sequence[float],
    ) -> Mapping[str, Answer]:
        answers: Dict[str, Answer] = {}
        for q, slot_indices in zip(questions, slots):
            match q:
                case Choice(name=name, options=opts):
                    keys = list(opts.keys()) if isinstance(opts, dict) else list(opts)
                    probs = sp.softmax([scores[i] for i in slot_indices])
                    best = int(probs.argmax())
                    answers[name] = ChoiceResult(
                        choice=keys[best],
                        confidence=float(probs[best]),
                        probabilities={k: float(p) for k, p in zip(keys, probs)},
                    )
                case Noul(name=name):
                    prob = float(sp.expit(scores[slot_indices[0]]))
                    answers[name] = NoulResult(is_positive=prob >= 0.5, probability=prob)
                case Score(name=name):
                    norm = float(sp.expit(scores[slot_indices[0]]))
                    answers[name] = ScoreResult(score=q.from_norm(norm), normalized=norm)
        return answers
