from dataclasses import dataclass
import json
from typing import Mapping, Protocol, Sequence, runtime_checkable

DecisionState = str | Mapping[str, str]

@runtime_checkable
class IAnswer(Protocol):
    """Polymorphic Data Stream Contract: Shared capability for typed answer values."""
    @property
    def value(self) -> str | bool | float: ...

AnswerInput = IAnswer | str | int | float | bool | None

@dataclass(frozen=True, slots=True)
class Choice:
    name: str
    options: Sequence[str] | Mapping[str, str]
    instruction: str | None = None
    target: str | int | None = None

    @property
    def target_index(self) -> int:
        if self.target is None: return 0
        if isinstance(self.target, int): return self.target
        keys = list(self.options.keys()) if isinstance(self.options, dict) else list(self.options)
        if self.target in keys: return keys.index(self.target)
        return int(self.target) if str(self.target).isdigit() else 0

    def shuffle(self) -> "Choice":
        import random
        keys = list(self.options.keys()) if isinstance(self.options, dict) else list(self.options)
        indices = list(range(len(keys)))
        random.shuffle(indices)
        target = self.target_index
        new_target = indices.index(target) if 0 <= target < len(indices) else 0
        opts = {keys[i]: self.options[keys[i]] for i in indices} if isinstance(self.options, dict) else [keys[i] for i in indices]
        return Choice(name=self.name, options=opts, instruction=self.instruction, target=new_target)

    def matches(self, answer: AnswerInput) -> bool:
        if answer is None or self.target is None: return False
        val = str(getattr(answer, "value", answer))
        if val == str(self.target): return True
        keys = list(self.options.keys()) if isinstance(self.options, dict) else list(self.options)
        if str(self.target).isdigit() and 0 <= int(self.target) < len(keys):
            return val == str(keys[int(self.target)])
        if val in keys and str(self.target) in keys:
            return keys.index(val) == keys.index(str(self.target))
        return False

@dataclass(frozen=True, slots=True)
class Noul:
    name: str
    instruction: str | None = None
    target: bool | float | int | None = None

    def matches(self, answer: AnswerInput) -> bool:
        if answer is None or self.target is None: return False
        val = getattr(answer, "value", answer)
        return bool(val) == bool(self.target)

@dataclass(frozen=True, slots=True)
class Score:
    name: str
    min_value: float = 0.0
    max_value: float = 100.0
    instruction: str | None = None
    target: float | None = None

    def matches(self, answer: AnswerInput) -> bool:
        if answer is None or self.target is None: return False
        val = float(getattr(answer, "value", answer))
        return abs(val - float(self.target)) < 1e-3

Question = Choice | Noul | Score

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
