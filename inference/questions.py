from dataclasses import dataclass
import json
from typing import Mapping, Protocol, Sequence, runtime_checkable

DecisionState = str | Mapping[str, str]

@dataclass(frozen=True, slots=True)
class Choice:
    name: str
    options: Sequence[str] | Mapping[str, str]
    instruction: str | None = None
    target: str | int | None = None

@dataclass(frozen=True, slots=True)
class Noul:
    name: str
    instruction: str | None = None
    target: bool | float | int | None = None

@dataclass(frozen=True, slots=True)
class Score:
    name: str
    min_value: float = 0.0
    max_value: float = 100.0
    instruction: str | None = None
    target: float | None = None

Question = Choice | Noul | Score

@runtime_checkable
class IAnswer(Protocol):
    """Polymorphic Data Stream Contract: Shared capability for typed answer values."""
    @property
    def value(self) -> str | bool | float: ...

@dataclass(frozen=True, slots=True)
class ChoiceResult:
    choice: str
    confidence: float
    probabilities: Mapping[str, float]
    @property
    def value(self) -> str: return self.choice

@dataclass(frozen=True, slots=True)
class NoulResult:
    is_positive: bool
    probability: float
    @property
    def value(self) -> bool: return self.is_positive

@dataclass(frozen=True, slots=True)
class ScoreResult:
    score: float
    normalized: float
    @property
    def value(self) -> float: return self.score

Answer = ChoiceResult | NoulResult | ScoreResult

@dataclass(slots=True)
class DecisionResult:
    answers: Mapping[str, Answer]
    latency_ms: float = 0.0

def format_state(state: DecisionState) -> str:
    """Colocated pure transform: formats state string or dictionary for token encoding."""
    if isinstance(state, dict): return " | ".join(f"{k}: {v}" for k, v in state.items())
    if isinstance(state, str) and state.startswith("{") and state.endswith("}"):
        try:
            parsed = json.loads(state)
            if isinstance(parsed, dict): return " | ".join(f"{k}: {v}" for k, v in parsed.items())
        except Exception: pass
    return str(state)
