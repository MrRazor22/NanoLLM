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
    decision: bool
    confidence: float

@dataclass(frozen=True, slots=True)
class ScoreAnswer(IAnswer):
    score: float
    confidence: float
    probabilities: Sequence[float] = ()

@runtime_checkable
class IOption(Protocol):
    answer: IAnswer | None

@dataclass(frozen=True, slots=True)
class ChoiceOption(IOption):
    options: Sequence[str]
    answer: ChoiceAnswer | None = None

@dataclass(frozen=True, slots=True)
class NoulOption(IOption):
    answer: NoulAnswer | None = None

@dataclass(frozen=True, slots=True)
class ScoreOption(IOption):
    criteria: Sequence[str] = ()
    answer: ScoreAnswer | None = None

def choice_keys(opt: ChoiceOption) -> Sequence[str]:
    return tuple(opt.options)

def choice_target_index(opt: ChoiceOption) -> int:
    if opt.answer is None:
        raise ValueError("ChoiceOption has no answer defined.")
    keys = choice_keys(opt)
    tgt = str(opt.answer.choice).strip()
    if tgt in keys: return keys.index(tgt)
    lower_keys = [k.lower() for k in keys]
    if tgt.lower() in lower_keys: return lower_keys.index(tgt.lower())
    if tgt.isdigit() and 0 <= int(tgt) < len(keys): return int(tgt)
    raise ValueError(f"Target '{tgt}' is not in options {keys}.")

def shuffle_choice(opt: ChoiceOption) -> ChoiceOption:
    import random
    keys = list(opt.options)
    random.shuffle(keys)
    return ChoiceOption(options=keys, answer=opt.answer)

def score_keys(opt: ScoreOption) -> Sequence[str]:
    return tuple(opt.criteria) if opt.criteria else ("0", "1", "2")

def score_target_index(opt: ScoreOption) -> int:
    if opt.answer is None:
        raise ValueError("ScoreOption has no answer defined.")
    keys = score_keys(opt)
    val = round(float(opt.answer.score))
    return max(0, min(len(keys) - 1, int(val)))

def matches_answer(gold_opt: IOption, pred_opt: IOption, tolerance: float = 0.5) -> bool:
    if gold_opt.answer is None or pred_opt.answer is None:
        return False
    match (gold_opt, pred_opt):
        case (ChoiceOption(options=opts), ChoiceOption(answer=pred_ans)) if gold_opt.answer is not None and pred_ans is not None:
            val, tgt = str(pred_ans.choice).strip(), str(gold_opt.answer.choice).strip()
            if val == tgt or val.lower() == tgt.lower(): return True
            keys = choice_keys(gold_opt)
            if tgt.isdigit() and 0 <= int(tgt) < len(keys):
                return val == str(keys[int(tgt)]) or val.lower() == str(keys[int(tgt)]).lower()
            return val in keys and tgt in keys and keys.index(val) == keys.index(tgt)
        case (NoulOption(answer=gold_ans), NoulOption(answer=pred_ans)) if gold_ans is not None and pred_ans is not None:
            return gold_ans.decision == pred_ans.decision
        case (ScoreOption(answer=gold_ans), ScoreOption(answer=pred_ans)) if gold_ans is not None and pred_ans is not None:
            return abs(pred_ans.score - gold_ans.score) <= tolerance
    return False
