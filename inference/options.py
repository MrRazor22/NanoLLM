from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable
import scipy.special as sp

# Sovereign High-Level Contracts: System One Options & Answers
@runtime_checkable
class IAnswer(Protocol):
    confidence: float

@dataclass(frozen=True, slots=True)
class ChoiceAnswer(IAnswer):
    choice: str
    confidence: float
    probabilities: Mapping[str, float]

@dataclass(frozen=True, slots=True)
class NoulAnswer(IAnswer):
    decision: bool
    confidence: float

@dataclass(frozen=True, slots=True)
class ScoreAnswer(IAnswer):
    score: float
    confidence: float
    probabilities: Sequence[float] = ()

@runtime_checkable
class IOption(Protocol):
    answer: IAnswer | None

@dataclass(frozen=True, slots=True)
class ChoiceOption(IOption):
    options: Sequence[str]
    answer: ChoiceAnswer | None = None

@dataclass(frozen=True, slots=True)
class NoulOption(IOption):
    answer: NoulAnswer | None = None

@dataclass(frozen=True, slots=True)
class ScoreOption(IOption):
    criteria: Sequence[str] = ()
    answer: ScoreAnswer | None = None

def option_to_candidates(option: IOption) -> Sequence[str]:
    match option:
        case ChoiceOption(options=opts):
            return tuple(opts.keys()) if isinstance(opts, dict) else tuple(opts)
        case ScoreOption(criteria=crit):
            return tuple(crit) if crit else ("0", "1", "2")
        case NoulOption():
            return ("false: does not hold", "true: holds")
        case _:
            raise ValueError(f"Unsupported option type: {type(option)}")

def resolve_decision(option: IOption, scores: Sequence[float]) -> IOption:
    match option:
        case ChoiceOption(options=opts):
            keys = tuple(opts.keys()) if isinstance(opts, dict) else tuple(opts)
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
        case _:
            raise ValueError(f"Unsupported option type: {type(option)}")
