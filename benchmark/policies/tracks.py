from pathlib import Path
from typing import Dict, Optional, Protocol, Sequence, runtime_checkable
from benchmark.scorecard import BenchmarkItem

from harness.training_dataset import load_raw_jsonl

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "data"

@runtime_checkable
class IBenchmarkSource(Protocol):
    name: str
    display_name: str
    def load(self, limit: Optional[int] = None) -> Sequence[BenchmarkItem]: ...

class JsonlBenchmarkSource:
    name: str = ""
    display_name: str = ""
    file_name: str = ""
    def load(self, limit: Optional[int] = None) -> Sequence[BenchmarkItem]:
        active_dir = LOCAL_BENCHMARK_DIR if LOCAL_BENCHMARK_DIR.exists() else BENCHMARK_DIR
        items = load_raw_jsonl(active_dir / self.file_name)
        return items[:limit] if limit is not None else items

class LayaSource(JsonlBenchmarkSource, IBenchmarkSource):
    name, display_name, file_name = "laya", "Laya Benchmark (421M SOTA Suite)", "laya.jsonl"

class AgenticSource(JsonlBenchmarkSource, IBenchmarkSource):
    name, display_name, file_name = "agentic", "Agentic Tool Routing", "agentic.jsonl"

class AbstentionSource(JsonlBenchmarkSource, IBenchmarkSource):
    name, display_name, file_name = "abstention", "Abstention & Safety", "abstention.jsonl"

class TypedDecisionsSource(JsonlBenchmarkSource, IBenchmarkSource):
    name, display_name, file_name = "typed_decisions", "Typed Decisions (Real-World)", "typed_decisions.jsonl"

TRACKS: Dict[str, type[IBenchmarkSource]] = {
    "agentic": AgenticSource, "abstention": AbstentionSource,
    "typed_decisions": TypedDecisionsSource, "laya": LayaSource,
}

__all__ = ["IBenchmarkSource", "LayaSource", "AgenticSource", "AbstentionSource", "TypedDecisionsSource", "TRACKS"]
