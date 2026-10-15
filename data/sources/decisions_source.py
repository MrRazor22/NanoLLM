import json
from typing import List, Sequence
from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

class DecisionsSource(IDataSource):
    """Real Typed-Decisions dataset extractor: loads multi-question typed decision benchmarks."""
    name = "typed_decisions"

    def __init__(self, repeat: int = 5):
        self.repeat = repeat

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("test", "test")]:
            rows = load_raw_jsonl(f"{self.name}_{split_name}", "LocalLLaMA/typed-decisions", split=split_name, subset="all")

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
