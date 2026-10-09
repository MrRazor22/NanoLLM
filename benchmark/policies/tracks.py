import json
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable
from nanollm import Choice
from benchmark.scorecard import BenchmarkItem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

@runtime_checkable
class IBenchmarkSource(Protocol):
    name: str
    display_name: str
    def load(self, limit: Optional[int] = None) -> Sequence[BenchmarkItem]: ...

def _load_benchmark_items(path: Path, limit: Optional[int] = None) -> List[BenchmarkItem]:
    if not path.exists():
        return []
    items: List[BenchmarkItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            qs = [
                Choice(
                    qid,
                    spec.get("options") or spec.get("criteria", {}),
                    instruction=spec.get("instructions"),
                )
                for qid, spec in raw.get("questions", {}).items()
            ]
            items.append(
                BenchmarkItem(
                    state=str(raw.get("state", "")),
                    questions=qs,
                    category=str(raw.get("category", "general")),
                    gold={k: str(v.get("label") if isinstance(v, dict) else v) for k, v in raw.get("gold", {}).items()},
                )
            )
            if limit and len(items) >= limit:
                break
    return items

class LayaSource(IBenchmarkSource):
    name = "laya"
    display_name = "Laya Benchmark (421M SOTA Suite)"
    def load(self, limit: Optional[int] = None) -> Sequence[BenchmarkItem]:
        return _load_benchmark_items(DATA_DIR / "laya.jsonl", limit)

class AgenticSource(IBenchmarkSource):
    name = "agentic"
    display_name = "Agentic Tool Routing"
    def load(self, limit: Optional[int] = None) -> Sequence[BenchmarkItem]:
        return _load_benchmark_items(DATA_DIR / "agentic.jsonl", limit)

class AbstentionSource(IBenchmarkSource):
    name = "abstention"
    display_name = "Abstention & Safety"
    def load(self, limit: Optional[int] = None) -> Sequence[BenchmarkItem]:
        return _load_benchmark_items(DATA_DIR / "abstention.jsonl", limit)

class TypedDecisionsSource(IBenchmarkSource):
    name = "typed_decisions"
    display_name = "Typed Decisions (Real-World)"
    def load(self, limit: Optional[int] = None) -> Sequence[BenchmarkItem]:
        return _load_benchmark_items(DATA_DIR / "typed_decisions.jsonl", limit)

TRACKS: Dict[str, type[IBenchmarkSource]] = {
    "agentic": AgenticSource,
    "abstention": AbstentionSource,
    "typed_decisions": TypedDecisionsSource,
    "laya": LayaSource,
}

__all__ = [
    "IBenchmarkSource",
    "LayaSource",
    "AgenticSource",
    "AbstentionSource",
    "TypedDecisionsSource",
    "TRACKS",
]
