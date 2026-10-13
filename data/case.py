from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence, runtime_checkable
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

@runtime_checkable
class IDataSource(Protocol):
    """Sovereign primitive contract: extracts domain records from external data origins."""
    name: str
    @property
    def signature(self) -> str: ...
    def read(self) -> Sequence[SourceRecord]: ...
