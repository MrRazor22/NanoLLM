import json
from pathlib import Path
from typing import Any, Dict, List
from datasets import load_dataset
from harness.dataset.training_dataset import ADAPTED_DIR, RAW_DIR, IDataSource

class TypedDecisionsSource(IDataSource):
    name = "typed_decisions"
    def __init__(self, repeat: int = 3):
        self.repeat = repeat

    def extract(self) -> List[Dict[str, Any]]:
        adapted_path = ADAPTED_DIR / f"{self.name}.jsonl"
        if adapted_path.exists():
            records = []
            with open(adapted_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records

        raw_path = RAW_DIR / f"{self.name}.jsonl"
        if raw_path.exists():
            rows = []
            with open(raw_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            ds = rows
        else:
            ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as f:
                for row in ds:
                    f.write(json.dumps(row, default=str) + "\n")

        records = []
        for row in ds:
            q_defs = row["questions"]
            g_defs = row["gold"]
            q_dict = json.loads(q_defs) if isinstance(q_defs, str) else q_defs
            g_dict = json.loads(g_defs) if isinstance(g_defs, str) else g_defs
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
        final_records = records * self.repeat

        ADAPTED_DIR.mkdir(parents=True, exist_ok=True)
        with open(adapted_path, "w", encoding="utf-8") as f:
            for r in final_records:
                f.write(json.dumps(r) + "\n")
        return final_records

__all__ = ["TypedDecisionsSource"]
