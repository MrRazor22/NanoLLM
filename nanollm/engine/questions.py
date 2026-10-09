from dataclasses import dataclass
import json
from typing import Mapping, Optional, Sequence, Union

# ==========================================
# Pure Passive DTOs (Zero Framework Types, Zero Any)
# ==========================================

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

@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    confidence: float
    probabilities: Mapping[str, float]

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
    answers: Mapping[str, Answer]
    latency_ms: float = 0.0

@dataclass(frozen=True)
class SlotIndices:
    positions: Sequence[int]

@dataclass(frozen=True)
class TokenLayout:
    input_ids: Sequence[int]
    mask: Sequence[float]
    slots: Sequence[SlotIndices]

    @property
    def slot_indices(self) -> Sequence[Sequence[int]]:
        return [s.positions for s in self.slots]

@dataclass(frozen=True)
class ScoreLayout:
    scores: Sequence[float]

# Backwards-compatible alias for unified schema
QuestionSpec = Question

@dataclass(frozen=True)
class DecisionSample:
    state: DecisionState
    questions: Sequence[Question]

# ==========================================
# Colocated Pure DTO Transforms
# ==========================================

def format_state(state: DecisionState) -> str:
    """Colocated pure transform: formats state string or dictionary for token encoding."""
    if isinstance(state, dict):
        return " | ".join(f"{k}: {v}" for k, v in state.items())
    if isinstance(state, str) and state.startswith("{") and state.endswith("}"):
        try:
            parsed = json.loads(state)
            if isinstance(parsed, dict):
                return " | ".join(f"{k}: {v}" for k, v in parsed.items())
        except Exception:
            pass
    return str(state)

__all__ = [
    "DecisionState",
    "Choice",
    "Noul",
    "Score",
    "Question",
    "ChoiceResult",
    "NoulResult",
    "ScoreResult",
    "Answer",
    "DecisionResult",
    "TokenLayout",
    "ScoreLayout",
    "QuestionSpec",
    "DecisionSample",
    "format_state",
]
