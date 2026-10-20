from dataclasses import dataclass
from typing import Mapping, Sequence

@dataclass(frozen=True, slots=True)
class DecisionCase:
    state: str
    candidates: Sequence[str]
    probabilities: Mapping[str, float]
    category: str
    source: str
    split: str | None = None

    def choice(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__) if self.probabilities else ""

    def target_index(self) -> int:
        c = self.choice()
        return self.candidates.index(c) if c in self.candidates else 0

    def to_dict(self) -> dict:
        res = {
            "state": self.state,
            "candidates": list(self.candidates),
            "probabilities": dict(self.probabilities),
            "category": self.category,
            "source": self.source,
        }
        if self.split is not None:
            res["split"] = self.split
        return res

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionCase":
        return cls(
            state=str(d["state"]),
            candidates=tuple(d["candidates"]),
            probabilities={str(k): float(v) for k, v in d["probabilities"].items()},
            category=str(d["category"]),
            source=str(d["source"]),
            split=d.get("split"),
        )

@dataclass(frozen=True, slots=True)
class DataSplits:
    train: Sequence[DecisionCase] = ()
    val: Sequence[DecisionCase] = ()
    test: Sequence[DecisionCase] = ()
