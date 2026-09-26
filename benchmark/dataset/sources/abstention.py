import json
from typing import Any, Dict, List
from benchmark.dataset.benchmark_dataset import ADAPTED_DIR, RAW_DIR, IDataSource

class AbstentionSource(IDataSource):
    name = "abstention"
    display_name = "2. Abstention (Out-of-Scope)"
    category_labels: Dict[str, str] = {
        "slice_missing_option": "Missing Option Abstention",
        "slice_distant_oos": "Distant Out-of-Scope",
    }

    def extract(self) -> List[Dict[str, Any]]:
        items = []
        for fn in ("slice_missing_option.jsonl", "slice_distant_oos.jsonl"):
            fp = RAW_DIR / fn
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
        return items

__all__ = ["AbstentionSource"]
