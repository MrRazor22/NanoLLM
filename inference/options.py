from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

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
    criteria: Sequence[str] | Mapping[str, str] = ("false", "true")

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
            if isinstance(crit, Mapping):
                return (crit.get("false", "false"), crit.get("true", "true"))
            return tuple(crit)
        case ScoreOption(criteria=crit):
            if len(crit) < 2:
                raise ValueError("ScoreOption requires at least 2 ordered criteria levels.")
            return tuple(crit)
        case _:
            raise ValueError(f"Unsupported option type: {type(option)}")

def probabilities_to_answer(option: IOption, probs: Sequence[float]) -> IAnswer:
    match option:
        case ChoiceOption(options=opts):
            keys = tuple(opts.keys()) if isinstance(opts, Mapping) else tuple(opts)
            best = int(max(range(len(probs)), key=lambda i: probs[i]))
            return ChoiceAnswer(
                choice=keys[best],
                confidence=float(probs[best]),
                probabilities={k: float(p) for k, p in zip(keys, probs)},
            )
        case NoulOption():
            p_true = float(probs[1]) if len(probs) >= 2 else float(probs[0])
            decision = p_true >= 0.5
            return NoulAnswer(
                noul=p_true,
                decision=decision,
                confidence=p_true if decision else 1.0 - p_true,
            )
        case ScoreOption():
            expected = float(sum(i * p for i, p in enumerate(probs)))
            return ScoreAnswer(score=expected, confidence=float(max(probs)), probabilities=list(probs))
        case _:
            raise ValueError(f"Unsupported option type: {type(option)}")
