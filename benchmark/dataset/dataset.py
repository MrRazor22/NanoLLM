import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

DATA_DIR = Path(__file__).resolve().parent / "data"

from training.dataset.dataset import IDataSource

class IBenchmarkDataset(Protocol):
    """Primitive contract for benchmark dataset loading."""
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]: ...

class BenchmarkDataset(IBenchmarkDataset):
    """Benchmark dataset primitive managing item loading and slicing."""
    def __init__(self, source: IDataSource, data_dir: Optional[Path] = None):
        self.source = source
        self.data_dir = data_dir or DATA_DIR
        self.dataset_path = self.data_dir / f"{self.source.name}.json"

    @property
    def display_name(self) -> str:
        return getattr(self.source, "display_name", self.source.name.replace("_", " ").title())

    @property
    def category_labels(self) -> Dict[str, str]:
        return getattr(self.source, "category_labels", {})

    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.dataset_path.exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data[:limit] if limit else data

        items = self.source.extract()
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        return items[:limit] if limit else items

__all__ = ["IDataSource", "IBenchmarkDataset", "BenchmarkDataset", "DATA_DIR"]
