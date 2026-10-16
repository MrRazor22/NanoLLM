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
        if self.target is None:
            raise ValueError(f"Choice question '{self.name}' has no target defined.")
        keys = list(self.options.keys()) if isinstance(self.options, dict) else list(self.options)
        if isinstance(self.target, int):
            if 0 <= self.target < len(keys):
                return self.target
            raise ValueError(f"Target index {self.target} is out of bounds for options of length {len(keys)} in '{self.name}'.")
        tgt_str = str(self.target).strip()
        if tgt_str in keys:
            return keys.index(tgt_str)
        lower_keys = [k.lower() for k in keys]
        if tgt_str.lower() in lower_keys:
            return lower_keys.index(tgt_str.lower())
        if tgt_str.isdigit() and 0 <= int(tgt_str) < len(keys):
            return int(tgt_str)
        raise ValueError(f"Target '{self.target}' is not in options {keys} for question '{self.name}'.")

    def shuffle(self) -> "Choice":
        import random
        keys = list(self.options.keys()) if isinstance(self.options, dict) else list(self.options)
        indices = list(range(len(keys)))
        random.shuffle(indices)
        target = self.target_index if self.target is not None else None
        new_target = indices.index(target) if (target is not None and 0 <= target < len(indices)) else None
        opts = {keys[i]: self.options[keys[i]] for i in indices} if isinstance(self.options, dict) else [keys[i] for i in indices]
        return Choice(name=self.name, options=opts, instruction=self.instruction, target=new_target)

    def matches(self, answer: AnswerInput) -> bool:
        if answer is None or self.target is None: return False
        val = str(getattr(answer, "value", answer)).strip()
        tgt = str(self.target).strip()
        if val == tgt or val.lower() == tgt.lower(): return True
        keys = list(self.options.keys()) if isinstance(self.options, dict) else list(self.options)
        if tgt.isdigit() and 0 <= int(tgt) < len(keys):
            return val == str(keys[int(tgt)]) or val.lower() == str(keys[int(tgt)]).lower()
        if val in keys and tgt in keys:
            return keys.index(val) == keys.index(tgt)
        return False

@dataclass(frozen=True, slots=True)
class Noul:
    name: str
    instruction: str | None = None
    target: bool | float | int | str | None = None

    def matches(self, answer: AnswerInput) -> bool:
        if answer is None or self.target is None: return False
        val = getattr(answer, "value", answer)
        val_bool = val.strip().lower() in ("true", "1", "yes") if isinstance(val, str) else bool(val)
        tgt_bool = self.target.strip().lower() in ("true", "1", "yes") if isinstance(self.target, str) else bool(self.target)
        return val_bool == tgt_bool

@dataclass(frozen=True, slots=True)
class Score:
    name: str
    min_value: float = 0.0
    max_value: float = 100.0
    instruction: str | None = None
    target: float | None = None

    def to_norm(self, value: float) -> float:
        """Projects continuous score value to bounded [0, 1] interval."""
        span = self.max_value - self.min_value
        return max(0.0, min(1.0, (float(value) - self.min_value) / span)) if span > 0 else 0.0

    def from_norm(self, norm: float) -> float:
        """Projects bounded [0, 1] activation back to continuous score domain."""
        return self.min_value + max(0.0, min(1.0, float(norm))) * (self.max_value - self.min_value)

    def matches(self, answer: AnswerInput, tolerance: float = 0.05) -> bool:
        if answer is None or self.target is None: return False
        val = float(getattr(answer, "value", answer))
        return abs(self.to_norm(val) - self.to_norm(float(self.target))) <= tolerance

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
        except json.JSONDecodeError:
            pass
    return str(state)
