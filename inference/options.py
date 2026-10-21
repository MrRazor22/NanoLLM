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
    noul: float
    decision: bool
    confidence: float

@dataclass(frozen=True, slots=True)
class ScoreAnswer(IAnswer):
    score: float
    confidence: float
    probabilities: Sequence[float] = ()

@runtime_checkable
class IOption(Protocol):
    """Decision query specification: represents candidate domains for resolution."""
    instruction: str

@dataclass(frozen=True, slots=True)
class ChoiceOption(IOption):
    options: Sequence[str] | Mapping[str, str]
    instruction: str = ""

@dataclass(frozen=True, slots=True)
class NoulOption(IOption):
    instruction: str
    criteria: Mapping[str, str] | None = None

@dataclass(frozen=True, slots=True)
class ScoreOption(IOption):
    criteria: Sequence[str]
    instruction: str = ""

def option_to_candidates(option: IOption) -> Sequence[str]:
    match option:
        case ChoiceOption(options=opts):
            if isinstance(opts, Mapping):
                return tuple(f"{k}: {v}" if v else k for k, v in opts.items())
            return tuple(opts)
        case NoulOption(criteria=crit):
            if crit:
                val_f = str(crit.get("false", crit.get("no", "false")))
                val_t = str(crit.get("true", crit.get("yes", "true")))
                return (val_f, val_t)
            return ("false", "true")
        case ScoreOption(criteria=crit):
            if not crit or len(crit) < 2:
                raise ValueError("ScoreOption requires at least 2 ordered criteria levels.")
            return tuple(crit)
        case _:
            raise ValueError(f"Unsupported option type: {type(option)}")

def scores_to_answer(option: IOption, scores: Sequence[float]) -> IAnswer:
    match option:
        case ChoiceOption(options=opts):
            keys = tuple(opts.keys()) if isinstance(opts, dict) else tuple(opts)
            probs = sp.softmax(scores)
            best = int(probs.argmax())
            return ChoiceAnswer(
                choice=keys[best],
                confidence=float(probs[best]),
                probabilities={k: float(p) for k, p in zip(keys, probs)},
            )
        case NoulOption():
            if len(scores) >= 2:
                probs = sp.softmax(scores)
                p_true = float(probs[1])
            else:
                p_true = float(sp.expit(scores[0]))
            decision = p_true >= 0.5
            return NoulAnswer(
                noul=p_true,
                decision=decision,
                confidence=p_true if decision else 1.0 - p_true,
            )
        case ScoreOption(criteria=crit):
            probs = sp.softmax(scores)
            expected = float(sum(i * p for i, p in enumerate(probs)))
            return ScoreAnswer(score=expected, confidence=float(max(probs)), probabilities=list(probs))
        case _:
            raise ValueError(f"Unsupported option type: {type(option)}")
