from dataclasses import dataclass, field
from typing import Mapping, Optional, Self, Sequence

@dataclass(frozen=True, slots=True)
class CaseMeta:
    category: str = ""
    source: str = ""
    option_type: str = "choice"
    split: Optional[str] = None

@dataclass(frozen=True, slots=True)
class DecisionCase:
    state: str
    candidates: Sequence[str]
    instruction: str = ""
    probabilities: Mapping[str, float] = field(default_factory=dict)
    meta: CaseMeta = field(default_factory=CaseMeta)

    def __post_init__(self):
        object.__setattr__(self, "state", self.state.strip())
        object.__setattr__(self, "instruction", self.instruction.strip())
        object.__setattr__(self, "candidates", tuple(c.strip() for c in self.candidates))
        object.__setattr__(self, "probabilities", {k.strip(): float(v) for k, v in self.probabilities.items()})

    def choice(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__) if self.probabilities else ""

    def target_index(self) -> int:
        c = self.choice()
        return self.candidates.index(c) if c in self.candidates else 0

    def to_dict(self) -> dict:
        res = {
            "state": self.state,
            "instruction": self.instruction,
            "candidates": list(self.candidates),
            "probabilities": dict(self.probabilities),
            "category": self.meta.category,
            "source": self.meta.source,
            "option_type": self.meta.option_type,
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
                category=str(d.get("category", "")),
                source=str(d.get("source", "")),
                option_type=str(d.get("option_type", "choice")),
                split=d.get("split"),
            ),
        )

@dataclass(frozen=True, slots=True)
class DataSplits:
    train: Sequence[DecisionCase] = ()
    val: Sequence[DecisionCase] = ()
    test: Sequence[DecisionCase] = ()
