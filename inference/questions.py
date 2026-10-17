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

@dataclass(frozen=True, slots=True)
class Noul:
    name: str
    instruction: str | None = None
    target: bool | float | int | str | None = None

@dataclass(frozen=True, slots=True)
class Score:
    name: str
    min_value: float = 0.0
    max_value: float = 100.0
    instruction: str | None = None
    target: float | None = None

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

def choice_keys(q: Choice) -> Sequence[str]:
    """Pure domain transform: extracts option keys from sequence or mapping."""
    return tuple(q.options.keys()) if isinstance(q.options, dict) else tuple(q.options)

def choice_target_index(q: Choice) -> int:
    """Pure domain transform: resolves integer target index from option keys."""
    if q.target is None:
        raise ValueError(f"Choice question '{q.name}' has no target defined.")
    keys = choice_keys(q)
    if isinstance(q.target, int):
        if 0 <= q.target < len(keys): return q.target
        raise ValueError(f"Target index {q.target} is out of bounds for options of length {len(keys)} in '{q.name}'.")
    tgt = str(q.target).strip()
    if tgt in keys: return keys.index(tgt)
    lower_keys = [k.lower() for k in keys]
    if tgt.lower() in lower_keys: return lower_keys.index(tgt.lower())
    if tgt.isdigit() and 0 <= int(tgt) < len(keys): return int(tgt)
    raise ValueError(f"Target '{q.target}' is not in options {keys} for question '{q.name}'.")

def shuffle_choice(q: Choice) -> Choice:
    """Pure domain transform: permutes choice candidate order preserving target alignment."""
    import random
    keys = list(choice_keys(q))
    indices = list(range(len(keys)))
    random.shuffle(indices)
    target = choice_target_index(q) if q.target is not None else None
    new_target = indices.index(target) if (target is not None and 0 <= target < len(indices)) else None
    opts = {keys[i]: q.options[keys[i]] for i in indices} if isinstance(q.options, dict) else [keys[i] for i in indices]
    return Choice(name=q.name, options=opts, instruction=q.instruction, target=new_target)

def score_to_norm(q: Score, value: float) -> float:
    """Pure domain transform: projects continuous score value to bounded [0, 1] interval."""
    span = q.max_value - q.min_value
    return max(0.0, min(1.0, (float(value) - q.min_value) / span)) if span > 0 else 0.0

def score_from_norm(q: Score, norm: float) -> float:
    """Pure domain transform: projects bounded [0, 1] activation back to continuous score domain."""
    return q.min_value + max(0.0, min(1.0, float(norm))) * (q.max_value - q.min_value)

def matches_answer(q: Question, answer: AnswerInput, tolerance: float = 0.05) -> bool:
    """Pure domain transform: evaluates polymorphic answer match against question target."""
    if answer is None or q.target is None: return False
    match q:
        case Choice():
            val, tgt = str(getattr(answer, "value", answer)).strip(), str(q.target).strip()
            if val == tgt or val.lower() == tgt.lower(): return True
            keys = choice_keys(q)
            if tgt.isdigit() and 0 <= int(tgt) < len(keys):
                return val == str(keys[int(tgt)]) or val.lower() == str(keys[int(tgt)]).lower()
            return val in keys and tgt in keys and keys.index(val) == keys.index(tgt)
        case Noul():
            val = getattr(answer, "value", answer)
            val_bool = val.strip().lower() in ("true", "1", "yes") if isinstance(val, str) else bool(val)
            tgt_bool = q.target.strip().lower() in ("true", "1", "yes") if isinstance(q.target, str) else bool(q.target)
            return val_bool == tgt_bool
        case Score():
            val = float(getattr(answer, "value", answer))
            return abs(score_to_norm(q, val) - score_to_norm(q, float(q.target))) <= tolerance
    return False

def format_state(state: DecisionState) -> str:
    """Colocated pure transform: formats state string or dictionary for token encoding."""
    if isinstance(state, dict): return " | ".join(f"{k}: {v}" for k, v in state.items())
    if isinstance(state, str) and state.startswith("{") and state.endswith("}"):
        try:
            if isinstance(parsed := json.loads(state), dict):
                return " | ".join(f"{k}: {v}" for k, v in parsed.items())
        except json.JSONDecodeError:
            pass
    return str(state)
