import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from benchmark.tracks.base import ITrackPolicy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class AbstentionTrackPolicy(ITrackPolicy):
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = []
        for fn in ("slice_missing_option.jsonl", "slice_distant_oos.jsonl"):
            fp = DATA_DIR / fn
            if not fp.exists():
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    items.append({
                        "id": r.get("id", ""),
                        "category": fp.stem,
                        "state": r.get("text", ""),
                        "questions": {
                            "intent": {
                                "type": "choice",
                                "instructions": r.get("question", ""),
                                "criteria": {c["id"]: c["description"] for c in r.get("candidates", [])},
                            }
                        },
                        "gold": {"intent": {"label": r.get("target_id", "__insufficient_evidence__")}},
                    })
        return items[:limit] if limit else items
