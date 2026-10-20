from typing import Protocol, Sequence, runtime_checkable
import scipy.special as sp

from inference.options import (
    ChoiceAnswer, ChoiceOption, IOption, NoulAnswer, NoulOption,
    ScoreAnswer, ScoreOption,
)

@runtime_checkable
class IResolver(Protocol):
    """Injected Opinion Sub-Primitive: Decodes raw float scores into typed domain Option with Answer."""
    def resolve(self, option: IOption, scores: Sequence[float]) -> IOption: ...

class DecisionResolver(IResolver):
    """Numerically stable resolver: decodes raw float scores using industry-standard primitives."""

    def resolve(self, option: IOption, scores: Sequence[float]) -> IOption:
        match option:
            case ChoiceOption(options=opts):
                keys = list(opts.keys()) if isinstance(opts, dict) else list(opts)
                probs = sp.softmax(scores)
                best = int(probs.argmax())
                ans = ChoiceAnswer(
                    choice=keys[best],
                    confidence=float(probs[best]),
                    probabilities={k: float(p) for k, p in zip(keys, probs)},
                )
                return ChoiceOption(options=opts, answer=ans)
            case NoulOption():
                if len(scores) >= 2:
                    probs = sp.softmax(scores)
                    p_true = float(probs[1])
                else:
                    p_true = float(sp.expit(scores[0]))
                ans = NoulAnswer(decision=p_true >= 0.5, confidence=p_true if p_true >= 0.5 else 1.0 - p_true)
                return NoulOption(answer=ans)
            case ScoreOption(criteria=crit):
                probs = sp.softmax(scores)
                expected = float(sum(i * p for i, p in enumerate(probs)))
                ans = ScoreAnswer(score=expected, confidence=float(max(probs)), probabilities=list(probs))
                return ScoreOption(criteria=crit, answer=ans)
