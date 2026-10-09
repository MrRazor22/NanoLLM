from dataclasses import dataclass
import json
from typing import Mapping, Optional, Sequence, Union
from inference.questions import Choice, DecisionState, Noul, Question, Score

@dataclass(frozen=True)
class DecisionCase:
    """Core Domain DTO: Represents a situated decision problem with associated questions."""
    state: DecisionState
    questions: Sequence[Question]
    category: str = "general"

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "category": self.category,
            "questions": {
                q.name: {
                    "type": "choice" if isinstance(q, Choice) else ("score" if isinstance(q, Score) else "noul"),
                    "options": getattr(q, "options", None),
                    "instructions": getattr(q, "instruction", None),
                }
                for q in self.questions
            },
            "gold": {q.name: getattr(q, "target", None) for q in self.questions},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionCase":
        state, cat = d.get("state", ""), d.get("category", "general")
        qs, gold = d.get("questions", {}), d.get("gold", {}) or {}
        specs = []
        if isinstance(qs, dict):
            for name, spec in qs.items():
                opts, ins = spec.get("options") or spec.get("criteria", {}), spec.get("instructions")
                q_type = spec.get("type", "choice" if opts else "noul")
                tgt = gold.get(name) if isinstance(gold, dict) else None
                if q_type == "choice" and opts:
                    keys = list(opts.keys()) if isinstance(opts, dict) else list(opts)
                    specs.append(Choice(name=name, options=opts, instruction=ins, target=keys.index(str(tgt)) if str(tgt) in keys else tgt))
                elif q_type == "score":
                    specs.append(Score(name=name, instruction=ins, target=float(tgt) if tgt is not None else 0.0))
                else:
                    specs.append(Noul(name=name, instruction=ins, target=bool(tgt) if tgt is not None else False))
        return cls(state=state, questions=specs, category=cat)

# Backward alias
DecisionSample = DecisionCase

__all__ = ["DecisionCase", "DecisionSample"]
