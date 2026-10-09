import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

DATA_DIR = Path(__file__).resolve().parent / "data"

@runtime_checkable
class IBenchmarkSource(Protocol):
    name: str
    display_name: str
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]: ...

def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists(): return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
        return lines[:limit] if limit else lines

class LayaSource(IBenchmarkSource):
    name = "laya"
    display_name = "Laya Benchmark (421M SOTA Suite)"
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return _load_jsonl(DATA_DIR / "laya.jsonl", limit)

class AgenticSource(IBenchmarkSource):
    name = "agentic"
    display_name = "Agentic Tool Routing"
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return _load_jsonl(DATA_DIR / "agentic.jsonl", limit)

class AbstentionSource(IBenchmarkSource):
    name = "abstention"
    display_name = "Abstention & Safety"
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return _load_jsonl(DATA_DIR / "abstention.jsonl", limit)

class TypedDecisionsSource(IBenchmarkSource):
    name = "typed_decisions"
    display_name = "Typed Decisions (Real-World)"
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return _load_jsonl(DATA_DIR / "typed_decisions.jsonl", limit)

TRACKS: Dict[str, type[IBenchmarkSource]] = {
    "agentic": AgenticSource,
    "abstention": AbstentionSource,
    "typed_decisions": TypedDecisionsSource,
    "laya": LayaSource,
}

__all__ = ["IBenchmarkSource", "LayaSource", "AgenticSource", "AbstentionSource", "TypedDecisionsSource", "TRACKS"]
