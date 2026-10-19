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

@dataclass(frozen=True, slots=True)
class DataSplits:
    train: Sequence[DecisionCase] = ()
    val: Sequence[DecisionCase] = ()
    test: Sequence[DecisionCase] = ()

def case_choice(case: DecisionCase) -> str:
    return max(case.probabilities, key=case.probabilities.__getitem__)

def case_confidence(case: DecisionCase) -> float:
    return float(max(case.probabilities.values()))

def case_target_index(case: DecisionCase) -> int:
    return case.candidates.index(case_choice(case))

def case_to_dict(case: DecisionCase) -> dict:
    res = {
        "state": case.state,
        "candidates": list(case.candidates),
        "probabilities": dict(case.probabilities),
        "category": case.category,
        "source": case.source,
    }
    if case.split is not None:
        res["split"] = case.split
    return res

def case_from_dict(d: dict) -> DecisionCase:
    return DecisionCase(
        state=str(d["state"]),
        candidates=tuple(d["candidates"]),
        probabilities={str(k): float(v) for k, v in d["probabilities"].items()},
        category=str(d["category"]),
        source=str(d["source"]),
        split=d.get("split"),
    )
