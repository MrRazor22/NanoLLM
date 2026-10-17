from typing import List, Sequence
from data.case import SourceRecord, case_from_dict
from data.sources.source import IDataSource
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
            split_records = [SourceRecord(sample=case_from_dict(r), tag=tag) for r in rows]
            records.extend(split_records * self.repeat if tag == "train" else split_records)
        return records
