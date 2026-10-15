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

    @property
    def gold(self) -> Mapping[str, str | int | float | bool]:
        return {q.name: q.target for q in self.questions if q.target is not None}

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "category": self.category,
            "questions": {
                q.name: {
                    "type": "choice" if isinstance(q, Choice) else ("score" if isinstance(q, Score) else "noul"),
                    "options": q.options if isinstance(q, Choice) else None,
                    "instructions": q.instruction,
                }
                for q in self.questions
            },
            "gold": dict(self.gold),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionCase":
        import json
        state = d.get("state", "")
        cat = d.get("category") or d.get("workflow", "general")
        raw_qs, raw_gold = d.get("questions", {}), d.get("gold", {}) or {}
        qs = json.loads(raw_qs) if isinstance(raw_qs, str) else (raw_qs or {})
        gold = json.loads(raw_gold) if isinstance(raw_gold, str) else (raw_gold or {})

        specs: list[Question] = []
        if isinstance(qs, dict):
            for name, spec in qs.items():
                crit = spec.get("criteria") or spec.get("options")
                ins = spec.get("instructions") or spec.get("instruction")
                q_type = spec.get("type", "choice" if crit else "noul")
                g_item = gold.get(name, {}) if isinstance(gold, dict) else None
                tgt = g_item.get("label") if isinstance(g_item, dict) else (g_item if g_item is not None else None)

                if q_type == "choice":
                    opts = crit if isinstance(crit, dict) else {o: o for o in (crit or [])}
                    specs.append(Choice(name=name, options=opts, instruction=ins, target=tgt))
                elif q_type == "noul":
                    opts = crit if (isinstance(crit, dict) and crit) else {"false": "False", "true": "True"}
                    specs.append(Choice(name=name, options=opts, instruction=ins, target=str(tgt).lower() if tgt is not None else None))
                elif q_type == "score" and isinstance(crit, list) and crit:
                    opts = {str(i): desc for i, desc in enumerate(crit)}
                    specs.append(Choice(name=name, options=opts, instruction=ins, target=str(tgt) if tgt is not None else None))
                elif q_type == "score":
                    min_v, max_v = float(spec.get("min_value", 0.0)), float(spec.get("max_value", 100.0))
                    specs.append(Score(name=name, min_value=min_v, max_value=max_v, instruction=ins, target=float(tgt) if tgt is not None else None))
                elif crit:
                    specs.append(Choice(name=name, options=crit, instruction=ins, target=tgt))
                else:
                    specs.append(Noul(name=name, instruction=ins, target=tgt))

        return cls(state=state, questions=specs, category=cat)

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
