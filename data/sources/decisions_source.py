from dataclasses import replace
import json
from typing import Sequence
from data.case import DecisionCase
from data.sources.source import IDataSource
from data.sources.source import load_raw_jsonl

class DecisionsSource(IDataSource):
    """Real Typed-Decisions dataset extractor: loads multi-question typed decision benchmarks."""
    name = "typed_decisions"

    def __init__(self, repeat: int = 5):
        self.repeat = repeat

    def cases(self) -> Sequence[DecisionCase]:
        train_rows = load_raw_jsonl(f"{self.name}_train", "LocalLLaMA/typed-decisions", split="train", subset="all")
        test_rows = load_raw_jsonl(f"{self.name}_test", "LocalLLaMA/typed-decisions", split="test", subset="all")

        train_cases = [replace(self._row_to_case(r), split="train") for r in train_rows] * self.repeat
        test_cases = [replace(self._row_to_case(r), split="test") for r in test_rows]
        return train_cases + test_cases

    def _row_to_case(self, d: dict) -> DecisionCase:
        state = str(d.get("state", ""))
        cat = str(d.get("category") or d.get("workflow") or "decision")
        qs_raw = d.get("questions")
        qs = json.loads(qs_raw) if isinstance(qs_raw, str) else qs_raw
        gold_raw = d.get("gold")
        gold = json.loads(gold_raw) if isinstance(gold_raw, str) else (gold_raw or {})
        _, spec = next(iter(qs.items()))
        crit = spec.get("options") or spec.get("criteria")
        if isinstance(crit, dict):
            candidates = tuple(str(o) for o in crit.keys())
        elif isinstance(crit, list) and crit:
            candidates = tuple(str(c) for c in crit)
        else:
            candidates = ("false", "true")

        g_item = next(iter(gold.values())) if isinstance(gold, dict) and gold else gold
        if isinstance(g_item, dict) and "probabilities" in g_item:
            raw_probs = g_item["probabilities"]
            probs = {c: float(raw_probs.get(c if c in raw_probs else str(i), 0.0)) for i, c in enumerate(candidates)}
            if sum(probs.values()) == 0.0 and "label" in g_item:
                lbl = str(g_item["label"]).strip()
                if lbl.isdigit() and 0 <= int(lbl) < len(candidates) and lbl not in candidates:
                    lbl = candidates[int(lbl)]
                probs = {c: (1.0 if c == lbl or c.lower() == lbl.lower() else 0.0) for c in candidates}
        else:
            raw_lbl = (g_item.get("label") if isinstance(g_item, dict) else g_item) if g_item else spec.get("label")
            lbl = str(raw_lbl).strip()
            if lbl.isdigit() and 0 <= int(lbl) < len(candidates) and lbl not in candidates:
                lbl = candidates[int(lbl)]
            probs = {c: (1.0 if c == lbl or c.lower() == lbl.lower() else 0.0) for c in candidates}

        if sum(probs.values()) == 0.0:
            raise ValueError(f"Target '{lbl}' not found in candidates {candidates}")

        return DecisionCase(
            state=state,
            candidates=candidates,
            probabilities=probs,
            category=cat,
            source=self.name,
        )
