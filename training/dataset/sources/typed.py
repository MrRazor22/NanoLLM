import json
from typing import Any, Dict, List
from datasets import load_dataset
from training.dataset.dataset import IDataSourcePolicy

class TypedDecisionsSource(IDataSourcePolicy):
    def __init__(self, repeat: int = 3):
        self.repeat = repeat

    def extract(self) -> List[Dict[str, Any]]:
        ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
        records = []
        for row in ds:
            q_dict, g_dict = json.loads(row["questions"]), json.loads(row["gold"])
            qs = []
            for qid, q_data in q_dict.items():
                if q_data["type"] == "choice":
                    qs.append({
                        "id": qid,
                        "instructions": q_data.get("instructions", "Select the best option:"),
                        "options": q_data["criteria"],
                        "gold": g_dict[qid]["label"],
                    })
            if qs:
                records.append({
                    "state": row["state"],
                    "questions": {q["id"]: {"type": "choice", "instructions": q["instructions"], "criteria": q["options"]} for q in qs},
                    "gold": {q["id"]: {"type": "choice", "label": q["gold"]} for q in qs},
                })
        return records * self.repeat

__all__ = ["TypedDecisionsSource"]
