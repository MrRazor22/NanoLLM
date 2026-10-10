from pathlib import Path
from typing import Callable, List, Sequence
from inference.case import DecisionCase
from data.sources.source import IDataSource, SourceRecord

DATA_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DATA_DIR / "raw"
ADAPTED_DIR = DATA_DIR / "adapted"

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

__all__ = [
    "DATA_DIR",
    "RAW_DIR",
    "ADAPTED_DIR",
    "IDataSource",
    "SourceRecord",
    "cached_jsonl",
    "GlaiveSource",
    "DecisionsSource",
    "GenericSource",
]
