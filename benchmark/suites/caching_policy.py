import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from benchmark.suites.base_policy import ISuitePolicy

class CachingSuitePolicy(ISuitePolicy):
    def __init__(self, inner: ISuitePolicy, cache_path: Path):
        self.inner = inner
        self.cache_path = Path(cache_path)

    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data[:limit] if limit else data

        items = self.inner.load(limit=None)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        return items[:limit] if limit else items

# Backward compatibility alias
CachingTrackPolicy = CachingSuitePolicy

__all__ = ["CachingSuitePolicy", "CachingTrackPolicy"]
