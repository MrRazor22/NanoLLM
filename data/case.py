from dataclasses import dataclass, field
from typing import Mapping, Optional, Self, Sequence

@dataclass(frozen=True, slots=True)
class CaseMeta:
    category: str
    source: str
    candidate_type: str
    split: Optional[str] = None

@dataclass(slots=True)
class DecisionCase:
    state: str
    candidates: Sequence[str]
    instruction: str = ""
    probabilities: Mapping[str, float] = field(default_factory=dict)
    meta: Optional[CaseMeta] = None

    def __init__(
        self,
        state: str,
        candidates: Sequence[str],
        instruction: str = "",
        probabilities: Optional[Mapping[str, float]] = None,
        meta: Optional[CaseMeta] = None,
    ):
        self.state = state.strip()
        self.candidates = tuple(c.strip() for c in candidates)
        self.instruction = instruction.strip()
        self.probabilities = {k.strip(): float(v) for k, v in probabilities.items()} if probabilities else {}
        self.meta = meta

    def choice(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__) if self.probabilities else ""

    def target_index(self) -> int:
        c = self.choice()
        return self.candidates.index(c) if c in self.candidates else 0

    def with_split(self, split: str) -> "DecisionCase":
        if self.meta is None:
            raise ValueError("Cannot assign split to DecisionCase without CaseMeta.")
        return DecisionCase(
            state=self.state,
            candidates=self.candidates,
            instruction=self.instruction,
            probabilities=self.probabilities,
            meta=CaseMeta(
                category=self.meta.category,
                source=self.meta.source,
                candidate_type=self.meta.candidate_type,
                split=split,
            ),
        )

    def to_dict(self) -> dict:
        if self.meta is None:
            raise ValueError("DecisionCase must contain CaseMeta for serialization.")
        res = {
            "state": self.state,
            "instruction": self.instruction,
            "candidates": list(self.candidates),
            "probabilities": dict(self.probabilities),
            "category": self.meta.category,
            "source": self.meta.source,
            "candidate_type": self.meta.candidate_type,
        }
        if self.meta.split is not None:
            res["split"] = self.meta.split
        return res

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            state=str(d["state"]),
            candidates=tuple(d["candidates"]),
            instruction=str(d.get("instruction", "")),
            probabilities={str(k): float(v) for k, v in d["probabilities"].items()},
            meta=CaseMeta(
                category=str(d["category"]),
                source=str(d["source"]),
                candidate_type=str(d["candidate_type"]),
                split=d.get("split"),
            ),
        )
