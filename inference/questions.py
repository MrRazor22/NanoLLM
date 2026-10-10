from dataclasses import dataclass
import json
from typing import Callable, Mapping, Optional, Protocol, Sequence, TypeVar, Union, runtime_checkable

DecisionState = Union[str, Mapping[str, str]]

@dataclass(frozen=True)
class Choice:
    name: str
    options: Union[Sequence[str], Mapping[str, str]]
    instruction: Optional[str] = None
    clusters: Optional[Mapping[str, Mapping[str, str]]] = None
    target: Optional[Union[str, int]] = None

@dataclass(frozen=True)
class Noul:
    name: str
    instruction: Optional[str] = None
    target: Optional[Union[bool, float, int]] = None

@dataclass(frozen=True)
class Score:
    name: str
    min_value: float = 0.0
    max_value: float = 100.0
    instruction: Optional[str] = None
    target: Optional[float] = None

Question = Union[Choice, Noul, Score]

@runtime_checkable
class IAnswer(Protocol):
    """Polymorphic Data Stream Contract: Shared capability for typed answer values."""
    @property
    def value(self) -> Union[str, bool, float]: ...

@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    confidence: float
    probabilities: Mapping[str, float]
    @property
    def value(self) -> str: return self.choice

@dataclass(frozen=True)
class NoulResult:
    is_positive: bool
    probability: float
    @property
    def value(self) -> bool: return self.is_positive

@dataclass(frozen=True)
class ScoreResult:
    score: float
    normalized: float
    @property
    def value(self) -> float: return self.score

Answer = Union[ChoiceResult, NoulResult, ScoreResult]

@dataclass
class DecisionResult:
    answers: Mapping[str, Answer]
    latency_ms: float = 0.0

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Operational boundary for neural decision inference."""
    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult: ...

@dataclass(frozen=True)
class TokenLayout:
    input_ids: Sequence[int]
    mask: Sequence[float]
    slots: Sequence[Sequence[int]]
    @property
    def slot_indices(self) -> Sequence[Sequence[int]]: return self.slots

@dataclass(frozen=True)
class ScoreLayout:
    scores: Sequence[float]

T = TypeVar("T")

class Composable:
    """Layer decorator composition mixin (lambda: P -> P)."""
    def add(self: T, layer: Union[type, Callable[[T], T]]) -> T:
        return layer(self) if isinstance(layer, type) else (getattr(layer, "attach")(self) if hasattr(layer, "attach") else layer) # type: ignore
    def __or__(self: T, layer: Union[type, Callable[[T], T]]) -> T:
        return self.add(layer)

def format_state(state: DecisionState) -> str:
    """Colocated pure transform: formats state string or dictionary for token encoding."""
    if isinstance(state, dict): return " | ".join(f"{k}: {v}" for k, v in state.items())
    if isinstance(state, str) and state.startswith("{") and state.endswith("}"):
        try:
            parsed = json.loads(state)
            if isinstance(parsed, dict): return " | ".join(f"{k}: {v}" for k, v in parsed.items())
        except Exception: pass
    return str(state)

__all__ = [
    "DecisionState", "Choice", "Noul", "Score", "Question",
    "IAnswer", "ChoiceResult", "NoulResult", "ScoreResult", "Answer",
    "IDecisionEngine", "DecisionResult", "TokenLayout", "ScoreLayout",
    "Composable", "format_state",
]
