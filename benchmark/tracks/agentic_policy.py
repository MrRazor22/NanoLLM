import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from benchmark.tracks.base_policy import ITrackPolicy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class AgenticPolicy(ITrackPolicy):
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with open(DATA_DIR / "benchmark.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[:limit] if limit else data
