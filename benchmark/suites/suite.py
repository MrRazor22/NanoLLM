import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable
import httpx

from data.case import DecisionCase
from data.sources.huggingface_source import RAW_DIR

@runtime_checkable
class IBenchmarkSuite(Protocol):
    """Sovereign benchmark suite contract: provides evaluation cases, identity, and published baselines."""
    @property
    def name(self) -> str: ...

    @property
    def published_baselines(self) -> Mapping[str, float]: ...

    def load_cases(self) -> Sequence[DecisionCase]: ...

def fetch_cached_jsonl(url: str, filename: str) -> list[dict]:
    """Single shared fetch-and-cache primitive for remote benchmark JSONL datasets."""
    target = RAW_DIR / filename
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    resp = httpx.get(url, headers={"User-Agent": "Piqo-Benchmark/1.0"}, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    rows = [json.loads(line) for line in resp.text.splitlines() if line.strip()]

    if rows:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows
