from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

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
    instruction: str

@dataclass(frozen=True, slots=True)
class ChoiceOption(IOption):
    instruction: str
    criteria: Mapping[str, str]

@dataclass(frozen=True, slots=True)
class NoulOption(IOption):
    instruction: str
    criteria: Mapping[str, str] = field(default_factory=lambda: {"false": "false", "true": "true"})

@dataclass(frozen=True, slots=True)
class ScoreOption(IOption):
    instruction: str
    criteria: Sequence[str]

def extract_candidates(option: IOption) -> Sequence[str]:
    match option:
        case ChoiceOption(criteria=crit):
            candidates = tuple(f"{k}: {v}" if v else k for k, v in crit.items())
        case NoulOption(criteria=crit):
            candidates = (crit["false"], crit["true"])
        case ScoreOption(criteria=crit):
            candidates = tuple(crit)
        case _:
            raise ValueError(f"Unsupported option type: {type(option)}")
    if len(candidates) < 2:
        raise ValueError("Options require at least 2 candidates.")
    return candidates

def resolve_answer(option: IOption, probs: Sequence[float]) -> IAnswer:
    match option:
        case ChoiceOption(criteria=crit):
            keys = tuple(crit)
            best = max(range(len(probs)), key=probs.__getitem__)
            return ChoiceAnswer(
                choice=keys[best],
                confidence=float(probs[best]),
                probabilities=dict(zip(keys, map(float, probs))),
            )
        case NoulOption():
            p_true = float(probs[1])
            decision = p_true >= 0.5
            return NoulAnswer(
                noul=p_true,
                decision=decision,
                confidence=max(p_true, 1.0 - p_true),
            )
        case ScoreOption():
            expected = sum(i * p for i, p in enumerate(probs))
            return ScoreAnswer(
                score=float(expected),
                confidence=float(max(probs)),
                probabilities=list(map(float, probs)),
            )
        case _:
            raise ValueError(f"Unsupported option type: {type(option)}")
