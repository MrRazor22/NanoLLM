import json
from typing import Optional, Sequence

from data.case import DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

class TypedDecisionsSource(IDataSource):
    """Source adapter for LocalLLaMA/typed-decisions multi-decision benchmark dataset."""
    name = "typed_decisions"

    def __init__(
        self,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
        cache: Optional[IRawCachePolicy] = None,
    ):
        self.train_limit = train_limit
        self.test_limit = test_limit
        self.cache = cache or HFCache()

    def cases(self) -> Sequence[DecisionCase]:
        train_rows = self.cache.load("LocalLLaMA/typed-decisions", split="train", subset="all", limit=self.train_limit)
        test_rows = self.cache.load("LocalLLaMA/typed-decisions", split="test", subset="all", limit=self.test_limit)

        cases: list[DecisionCase] = []
        for split, rows in (("train", train_rows), ("test", test_rows)):
            for d in rows:
                state = str(d.get("state") or "")
                cat = str(d.get("category") or d.get("workflow") or "decision")
                qs = json.loads(d["questions"]) if isinstance(d.get("questions"), str) else (d.get("questions") or {})
                gold = json.loads(d["gold"]) if isinstance(d.get("gold"), str) else (d.get("gold") or {})

                for q_name, spec in qs.items():
                    g_item = gold.get(q_name) or {}
                    crit = spec.get("options") or spec.get("criteria")
                    if isinstance(crit, dict):
                        candidates = tuple(str(k) for k in crit.keys())
                    elif isinstance(crit, list) and crit:
                        candidates = tuple(str(c) for c in crit)
                    else:
                        candidates = ("false", "true")

                    raw_probs = g_item.get("probabilities") if isinstance(g_item, dict) else None
                    if raw_probs and isinstance(raw_probs, dict):
                        probs = {c: float(raw_probs.get(c, raw_probs.get(str(i), 0.0))) for i, c in enumerate(candidates)}
                    else:
                        lbl = str((g_item.get("label") if isinstance(g_item, dict) else g_item) or "").strip()
                        if lbl.isdigit() and 0 <= int(lbl) < len(candidates) and lbl not in candidates:
                            lbl = candidates[int(lbl)]
                        probs = {c: (1.0 if c == lbl or c.lower() == lbl.lower() else 0.0) for c in candidates}

                    if sum(probs.values()) == 0.0:
                        continue

                    instr = spec.get("instructions")
                    full_state = f"{state}\n{instr}" if instr and instr not in state else state
                    cases.append(
                        DecisionCase(
                            state=full_state,
                            candidates=candidates,
                            probabilities=probs,
                            category=cat,
                            source=self.name,
                            split=split,
                        )
                    )

        return cases

__all__ = ["TypedDecisionsSource"]
