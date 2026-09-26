import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class ISuitePolicy(Protocol):
    """Bedrock contract for any benchmark suite."""
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]: ...

class BaseSuitePolicy(ABC, ISuitePolicy):
    """Template method base: handles standardized disk caching so children only define extraction."""
    name: str

    @property
    def cache_path(self) -> Path:
        return DATA_DIR / f"{self.name}.cache.json"

    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data[:limit] if limit else data

        items = self.extract()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        return items[:limit] if limit else items

    @abstractmethod
    def extract(self) -> List[Dict[str, Any]]: ...

# Backward compatibility alias
ITrackPolicy = ISuitePolicy

__all__ = ["ISuitePolicy", "ITrackPolicy", "BaseSuitePolicy", "DATA_DIR"]
