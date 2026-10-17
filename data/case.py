from dataclasses import dataclass
from typing import Literal, Mapping, Sequence
from inference.questions import Choice, DecisionState, Noul, Question, Score

SplitTag = Literal["train", "val", "test"]

@dataclass(frozen=True, slots=True)
class DecisionCase:
    """Core Domain DTO: Represents a situated decision problem with associated questions."""
    state: DecisionState
    questions: Sequence[Question]
    category: str = "general"

def case_gold(case: DecisionCase) -> Mapping[str, str | int | float | bool]:
    """Pure domain transform: extracts target labels from case questions."""
    return {q.name: q.target for q in case.questions if q.target is not None}

def case_to_dict(case: DecisionCase) -> dict:
    """Pure domain transform: serializes DecisionCase DTO to dictionary with exact polymorphic types."""
    q_dict = {
        q.name: ({"type": "choice", "options": q.options, "instructions": q.instruction} if isinstance(q, Choice)
                 else ({"type": "score", "min_value": q.min_value, "max_value": q.max_value, "instructions": q.instruction} if isinstance(q, Score)
                       else {"type": "noul", "instructions": q.instruction}))
        for q in case.questions
    }
    gold = {q.name: q.target for q in case.questions if q.target is not None}
    return {"state": case.state, "category": case.category, "questions": q_dict, "gold": gold}

def case_from_dict(d: dict) -> DecisionCase:
    """Pure domain transform: deserializes dictionary to DecisionCase preserving exact Question variants."""
    import json
    state = d.get("state", "")
    cat = d.get("category") or d.get("workflow", "general")
    raw_qs, raw_gold = d.get("questions", {}), d.get("gold", {}) or {}
    qs = json.loads(raw_qs) if isinstance(raw_qs, str) else (raw_qs or {})
    gold = json.loads(raw_gold) if isinstance(raw_gold, str) else (raw_gold or {})

    specs: list[Question] = []
    if isinstance(qs, dict):
        for name, spec in qs.items():
            ins = spec.get("instructions") or spec.get("instruction")
            q_type = spec.get("type")
            g_item = gold.get(name) if isinstance(gold, dict) else None
            tgt = g_item.get("label") if isinstance(g_item, dict) else (g_item if g_item is not None else spec.get("label", spec.get("target")))

            if q_type == "score" or (q_type is None and "min_value" in spec):
                min_v = float(spec.get("min_value", 0.0))
                max_v = float(spec.get("max_value", 100.0))
                specs.append(Score(name=name, min_value=min_v, max_value=max_v, instruction=ins, target=float(tgt) if tgt is not None else None))
            elif q_type == "noul" or (q_type is None and not (spec.get("options") or spec.get("criteria"))):
                specs.append(Noul(name=name, instruction=ins, target=tgt))
            else:
                crit = spec.get("options") or spec.get("criteria")
                opts = crit if isinstance(crit, dict) else {o: o for o in (crit or [])}
                if not opts:
                    raise ValueError(f"Choice question '{name}' has empty options in serialized case.")
                specs.append(Choice(name=name, options=opts, instruction=ins, target=tgt))

    return DecisionCase(state=state, questions=specs, category=cat)

@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Core Domain DTO: Ingested decision case tagged with origin partition."""
    sample: DecisionCase
    tag: SplitTag = "train"

@dataclass(frozen=True, slots=True)
class DataSplits:
    """Core Domain DTO: Consolidated partitioned dataset splits."""
    train: Sequence[DecisionCase]
    val: Sequence[DecisionCase]
    test: Sequence[DecisionCase]
