import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from benchmark.suites.base_policy import ITrackPolicy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class AgenticPolicy(ITrackPolicy):
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        raw_path = DATA_DIR / "benchmark.json"
        with open(raw_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        return items[:limit] if limit else items
