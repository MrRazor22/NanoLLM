import json
from typing import Any, Dict, List
from benchmark.dataset import DATA_DIR, ISuiteSourcePolicy

class AgenticSource(ISuiteSourcePolicy):
    name = "agentic"

    def extract(self) -> List[Dict[str, Any]]:
        raw_path = DATA_DIR / "benchmark.json"
        with open(raw_path, "r", encoding="utf-8") as f:
            return json.load(f)

__all__ = ["AgenticSource"]
