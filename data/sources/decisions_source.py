import json
from pathlib import Path
from typing import List, Sequence
from datasets import load_dataset
from data.case import DecisionCase, IDataSource, SourceRecord
from inference.questions import Choice
from data.sources.huggingface_source import RAW_DIR, read_raw_jsonl

class DecisionsSource(IDataSource):
    """Real Typed-Decisions dataset extractor: loads multi-question typed decision benchmarks."""
    name = "typed_decisions"

    def __init__(self, repeat: int = 5):
        self.repeat = repeat

    @property
    def signature(self) -> str:
        import hashlib
        return hashlib.sha256(f"decisions_{self.repeat}".encode()).hexdigest()[:10]

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("test", "test")]:
            target = RAW_DIR / f"{self.name}_{split_name}.jsonl"
            rows = read_raw_jsonl(
                target,
                lambda s=split_name: list(load_dataset("LocalLLaMA/typed-decisions", "all", split=s)),
            )

            split_records: List[SourceRecord] = []
            for row in rows:
                q_dict = json.loads(q) if isinstance(q := row.get("questions", {}), str) else q
                g_dict = json.loads(g) if isinstance(g := row.get("gold", {}), str) else g
                qs = [
                    Choice(name=qid, options=qd.get("criteria", {}), instruction=qd.get("instructions", "Select the best option:"), target=g_dict.get(qid, {}).get("label"))
                    for qid, qd in q_dict.items() if qd.get("type") == "choice"
                ]
                if qs:
                    split_records.append(SourceRecord(sample=DecisionCase(state=row.get("state", ""), questions=qs, category=row.get("category", "typed_decisions")), tag=tag))
            if tag == "train":
                records.extend(split_records * self.repeat)
            else:
                records.extend(split_records)
        return records

__all__ = ["DecisionsSource"]
