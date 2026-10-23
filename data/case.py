from dataclasses import dataclass, field
from typing import Mapping, Optional, Self, Sequence

@dataclass(frozen=True, slots=True)
class CaseMeta:
    category: str
    source: str
    candidate_type: str
    split: Optional[str] = None

    def __post_init__(self):
        if not self.category or not self.source or not self.candidate_type:
            raise ValueError("CaseMeta requires non-empty category, source, and candidate_type.")

    @classmethod
    def runtime(cls, candidate_type: str) -> "CaseMeta":
        return cls(category="runtime", source="runtime", candidate_type=candidate_type)

@dataclass(slots=True)
class DecisionCase:
    state: str
    candidates: Sequence[str]
    instruction: str
    probabilities: Mapping[str, float]
    meta: CaseMeta

    def __init__(
        self,
        state: str,
        candidates: Sequence[str],
        instruction: str,
        probabilities: Mapping[str, float],
        meta: CaseMeta,
    ):
        if not isinstance(meta, CaseMeta):
            raise TypeError("DecisionCase requires a valid CaseMeta.")
        self.state = state.strip()
        self.candidates = tuple(c.strip() for c in candidates)
        self.instruction = instruction.strip()
        self.probabilities = {k.strip(): float(v) for k, v in probabilities.items()}
        self.meta = meta

    def option(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__) if self.probabilities else ""

    def target_index(self) -> int:
        opt = self.option()
        return self.candidates.index(opt) if opt in self.candidates else 0

    def with_split(self, split: str) -> "DecisionCase":
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
