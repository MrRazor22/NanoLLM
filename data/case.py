from dataclasses import dataclass
import json
from typing import Mapping, Protocol, Sequence, runtime_checkable

# Sovereign Domain Contracts: System One Options & Answers
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
    min_value: float = 0.0
    max_value: float = 100.0
    answer: ScoreAnswer | None = None

# Core Domain DTOs
@dataclass(frozen=True, slots=True)
class DecisionCase:
    state: str
    option: IOption
    category: str = "general"
    split: str | None = None
    source: str = ""

@dataclass(frozen=True, slots=True)
class DataSplits:
    train: Sequence[DecisionCase] = ()
    val: Sequence[DecisionCase] = ()
    test: Sequence[DecisionCase] = ()

# Pure Domain Transforms
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

def score_to_norm(opt: ScoreOption, value: float) -> float:
    span = opt.max_value - opt.min_value
    return max(0.0, min(1.0, (float(value) - opt.min_value) / span)) if span > 0 else 0.0

def score_from_norm(opt: ScoreOption, norm: float) -> float:
    return opt.min_value + max(0.0, min(1.0, float(norm))) * (opt.max_value - opt.min_value)

def matches_answer(gold_opt: IOption, pred_opt: IOption, tolerance: float = 0.05) -> bool:
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
            return abs(score_to_norm(gold_opt, pred_ans.score) - score_to_norm(gold_opt, gold_ans.score)) <= tolerance
    return False

def format_state(state: str) -> str:
    if state.startswith("{") and state.endswith("}"):
        try:
            if isinstance(parsed := json.loads(state), dict):
                return " | ".join(f"{k}: {v}" for k, v in parsed.items())
        except json.JSONDecodeError:
            pass
    return state

def case_to_dict(case: DecisionCase) -> dict:
    opt = case.option
    if isinstance(opt, ChoiceOption):
        opt_dict = {"type": "choice", "options": list(opt.options)}
        if opt.answer is not None:
            opt_dict["answer"] = opt.answer.choice
    elif isinstance(opt, ScoreOption):
        opt_dict = {"type": "score", "min_value": opt.min_value, "max_value": opt.max_value}
        if opt.answer is not None:
            opt_dict["answer"] = opt.answer.score
    else:
        opt_dict = {"type": "noul"}
        if opt.answer is not None:
            opt_dict["answer"] = opt.answer.decision

    res = {"state": case.state, "category": case.category, "option": opt_dict}
    if case.split is not None:
        res["split"] = case.split
    if case.source:
        res["source"] = case.source
    return res

def case_from_dict(d: dict) -> DecisionCase:
    state = str(d.get("state", ""))
    cat = d.get("category") or d.get("workflow", "general")
    split = d.get("split")
    source = d.get("source", "")

    opt_raw = d.get("option")
    if opt_raw is None:
        qs_raw = d.get("questions", {})
        qs = json.loads(qs_raw) if isinstance(qs_raw, str) else (qs_raw or {})
        gold_raw = d.get("gold", {}) or {}
        gold = json.loads(gold_raw) if isinstance(gold_raw, str) else (gold_raw or {})
        if isinstance(qs, dict) and qs:
            name, spec = next(iter(qs.items()))
            g_item = gold.get(name) if isinstance(gold, dict) else None
            tgt = g_item.get("label") if isinstance(g_item, dict) else (g_item if g_item is not None else spec.get("target", spec.get("label")))
            opt_raw = dict(spec)
            if tgt is not None:
                opt_raw["answer"] = tgt
        else:
            opt_raw = {}
    elif isinstance(opt_raw, str):
        opt_raw = json.loads(opt_raw)

    opt_type = opt_raw.get("type")
    ans_val = opt_raw.get("answer", opt_raw.get("target", opt_raw.get("label")))

    option: IOption
    if opt_type == "score" or (opt_type is None and "min_value" in opt_raw):
        min_v = float(opt_raw.get("min_value", 0.0))
        max_v = float(opt_raw.get("max_value", 100.0))
        ans = ScoreAnswer(score=float(ans_val), confidence=1.0) if ans_val is not None else None
        option = ScoreOption(min_value=min_v, max_value=max_v, answer=ans)
    elif opt_type == "noul" or (opt_type is None and not (opt_raw.get("options") or opt_raw.get("criteria"))):
        if ans_val is not None:
            is_pos = (ans_val is True or str(ans_val).lower() in ("true", "1", "yes"))
            ans = NoulAnswer(decision=is_pos, confidence=1.0)
        else:
            ans = None
        option = NoulOption(answer=ans)
    else:
        crit = opt_raw.get("options") or opt_raw.get("criteria")
        opts = list(crit.keys()) if isinstance(crit, dict) else list(crit or [])
        ans = ChoiceAnswer(choice=str(ans_val), confidence=1.0, probabilities={str(ans_val): 1.0}) if ans_val is not None else None
        option = ChoiceOption(options=opts, answer=ans)

    return DecisionCase(state=state, option=option, category=cat, split=split, source=source)
