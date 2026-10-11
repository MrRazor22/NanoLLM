from pathlib import Path
from typing import Callable, List
from data.sources.data_source import IDataSource, SourceRecord

DATA_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DATA_DIR / "raw"
ADAPTED_DIR = DATA_DIR / "adapted"

def cached_records(path: Path, fetcher: Callable[[], Sequence[SourceRecord]]) -> Sequence[SourceRecord]:
    import json
    from inference.case import DecisionCase
    if path.exists():
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                records.append(SourceRecord(sample=DecisionCase.from_dict(d), tag=d.get("tag", "unsplit")))
        return records
    records = list(fetcher())
    if records:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({**r.sample.to_dict(), "tag": r.tag}, default=str) for r in records]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records

def cached_jsonl(path: Path, fetcher: Callable[[], List[dict]]) -> List[dict]:
    import json
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = fetcher()
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows

from data.sources.glaive_source import GlaiveSource
from data.sources.decisions_source import DecisionsSource
from data.sources.generic_source import GenericSource
from data.sources.standard import standard_sources

__all__ = [
    "DATA_DIR",
    "RAW_DIR",
    "ADAPTED_DIR",
    "IDataSource",
    "SourceRecord",
    "cached_jsonl",
    "cached_records",
    "GlaiveSource",
    "DecisionsSource",
    "GenericSource",
    "standard_sources",
]
