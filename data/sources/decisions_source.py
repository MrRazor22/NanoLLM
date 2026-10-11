import json
from typing import List, Sequence
from datasets import load_dataset
from inference.case import DecisionCase
from inference.questions import Choice
from data.sources import RAW_DIR, ADAPTED_DIR, IDataSource, SourceRecord, cached_jsonl, cached_records

class DecisionsSource(IDataSource):
    """Real Typed-Decisions dataset extractor: loads multi-question typed decision benchmarks."""
    name = "typed_decisions"

    def __init__(self, repeat: int = 5):
        self.repeat = repeat

    def read(self) -> Sequence[SourceRecord]:
        def _build() -> List[SourceRecord]:
            rows = cached_jsonl(
                RAW_DIR / f"{self.name}.jsonl",
                lambda: list(load_dataset("LocalLLaMA/typed-decisions", "all", split="train")),
            )
            records: List[SourceRecord] = []
            for row in rows:
                q_dict = json.loads(q) if isinstance(q := row.get("questions", {}), str) else q
                g_dict = json.loads(g) if isinstance(g := row.get("gold", {}), str) else g
                qs = [
                    Choice(name=qid, options=qd.get("criteria", {}), instruction=qd.get("instructions", "Select the best option:"), target=g_dict.get(qid, {}).get("label"))
                    for qid, qd in q_dict.items() if qd.get("type") == "choice"
                ]
                if qs:
                    records.append(SourceRecord(sample=DecisionCase(state=row.get("state", ""), questions=qs, category=row.get("category", "typed_decisions")), tag="unsplit"))
            return records

        return cached_records(ADAPTED_DIR / f"{self.name}.jsonl", _build) * self.repeat

__all__ = ["DecisionsSource"]
