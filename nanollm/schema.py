from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass(frozen=True)
class Choice:
    name: str
    options: Union[List[str], Dict[str, str]]
    instruction: Optional[str] = None

@dataclass(frozen=True)
class Noul:
    name: str
    instruction: Optional[str] = None

@dataclass(frozen=True)
class Score:
    name: str
    min_value: float = 0.0
    max_value: float = 100.0
    instruction: Optional[str] = None

Question = Union[Choice, Noul, Score]

@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    confidence: float
    probabilities: Dict[str, float]

@dataclass(frozen=True)
class NoulResult:
    value: bool
    probability: float

@dataclass(frozen=True)
class ScoreResult:
    score: float
    normalized: float

Answer = Union[ChoiceResult, NoulResult, ScoreResult]

@dataclass
class DecisionResult:
    answers: Dict[str, Answer]
    latency_ms: float = 0.0
