from typing import Any, Dict, List, Optional
from benchmark.dataset import BenchmarkDataset, DATA_DIR, IBenchmarkDataset, ISuiteSourcePolicy

class BaseSuitePolicy(ISuiteSourcePolicy):
    """Convenience base suite defining a dataset-backed benchmark policy."""
    name: str

    def extract(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return BenchmarkDataset(self).load(limit=limit)

__all__ = ["BaseSuitePolicy", "ISuiteSourcePolicy", "IBenchmarkDataset", "BenchmarkDataset", "DATA_DIR"]
