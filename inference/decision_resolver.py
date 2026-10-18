from typing import Protocol, Sequence, runtime_checkable
import scipy.special as sp
from data.case import (
    ChoiceAnswer, ChoiceOption, IOption, NoulAnswer, NoulOption,
    ScoreAnswer, ScoreOption, score_from_norm,
)

@runtime_checkable
class IResolver(Protocol):
    """Injected Opinion Sub-Primitive: Decodes raw float scores into typed domain Option with Answer."""
    def resolve(
        self,
        option: IOption,
        slots: Sequence[Sequence[int]],
        scores: Sequence[float],
    ) -> IOption:
        ...

class DecisionResolver(IResolver):
    """Numerically stable resolver: decodes raw float scores using industry-standard primitives."""

    def resolve(
        self,
        option: IOption,
        slots: Sequence[Sequence[int]],
        scores: Sequence[float],
    ) -> IOption:
        slot_indices = slots[0]
        match option:
            case ChoiceOption(options=opts):
                keys = list(opts.keys()) if isinstance(opts, dict) else list(opts)
                probs = sp.softmax([scores[i] for i in slot_indices])
                best = int(probs.argmax())
                ans = ChoiceAnswer(
                    choice=keys[best],
                    confidence=float(probs[best]),
                    probabilities={k: float(p) for k, p in zip(keys, probs)},
                )
                return ChoiceOption(options=opts, answer=ans)
            case NoulOption():
                prob = float(sp.expit(scores[slot_indices[0]]))
                ans = NoulAnswer(decision=prob >= 0.5, confidence=prob if prob >= 0.5 else 1.0 - prob)
                return NoulOption(answer=ans)
            case ScoreOption(min_value=min_v, max_value=max_v):
                norm = float(sp.expit(scores[slot_indices[0]]))
                ans = ScoreAnswer(score=score_from_norm(option, norm), confidence=1.0)
                return ScoreOption(min_value=min_v, max_value=max_v, answer=ans)
