from dataclasses import dataclass, field
from typing import Mapping, Optional, Self, Sequence

@dataclass(frozen=True, slots=True)
class CaseMeta:
    category: str = ""
    source: str = ""
    split: str | None = None

@dataclass(frozen=True, slots=True)
class DecisionCase:
    state: str
    candidates: Sequence[str]
    instruction: str = ""
    probabilities: Mapping[str, float] = field(default_factory=dict)
    meta: CaseMeta = field(default_factory=CaseMeta)

    def __init__(
        self,
        state: str,
        candidates: Sequence[str],
        instruction: str = "",
        probabilities: Optional[Mapping[str, float]] = None,
        meta: Optional[CaseMeta] = None,
        category: str = "",
        source: str = "",
        split: Optional[str] = None,
    ):
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "candidates", candidates if isinstance(candidates, tuple) else tuple(candidates))
        object.__setattr__(self, "instruction", instruction)
        object.__setattr__(self, "probabilities", probabilities or {})
        resolved_meta = meta or CaseMeta(category=category, source=source, split=split)
        object.__setattr__(self, "meta", resolved_meta)

    @property
    def category(self) -> str:
        return self.meta.category

    @property
    def source(self) -> str:
        return self.meta.source

    @property
    def split(self) -> str | None:
        return self.meta.split

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
            "category": self.meta.category,
            "source": self.meta.source,
        }
        if self.meta.split is not None:
            res["split"] = self.meta.split
        return res

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            state=str(d["state"]),
            candidates=tuple(d["candidates"]),
            probabilities={str(k): float(v) for k, v in d["probabilities"].items()},
            meta=CaseMeta(
                category=str(d.get("category", "")),
                source=str(d.get("source", "")),
                split=d.get("split"),
            ),
        )

@dataclass(frozen=True, slots=True)
class DataSplits:
    train: Sequence[DecisionCase] = ()
    val: Sequence[DecisionCase] = ()
    test: Sequence[DecisionCase] = ()
