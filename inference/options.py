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
