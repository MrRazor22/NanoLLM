import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

DATA_DIR = Path(__file__).resolve().parent / "data"

class ISuiteSourcePolicy(Protocol):
    """Pure strategy policy defining how a benchmark suite extracts its raw evaluation items."""
    name: str
    def extract(self) -> List[Dict[str, Any]]: ...

class IBenchmarkDataset(Protocol):
    """Primitive contract for benchmark dataset loading."""
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]: ...

class BenchmarkDataset(IBenchmarkDataset):
    """Benchmark dataset primitive managing item loading, disk caching, and slicing."""
    def __init__(self, policy: ISuiteSourcePolicy, cache_dir: Optional[Path] = None):
        self.policy = policy
        self.cache_dir = cache_dir or DATA_DIR
        self.cache_path = self.cache_dir / f"{self.policy.name}.cache.json"

    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data[:limit] if limit else data

        items = self.policy.extract()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        return items[:limit] if limit else items

__all__ = ["ISuiteSourcePolicy", "IBenchmarkDataset", "BenchmarkDataset", "DATA_DIR"]
