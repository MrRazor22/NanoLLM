import json
from pathlib import Path
from typing import Any, Dict, List
from datasets import load_dataset
from harness.dataset.training_dataset import ADAPTED_DIR, RAW_DIR, IDataSource
from harness.dataset.transforms import load_raw_jsonl, save_jsonl

class TypedDecisionsSource(IDataSource):
    name = "typed_decisions"
    def __init__(self, repeat: int = 3):
        self.repeat = repeat

    def extract(self) -> List[Dict[str, Any]]:
        adapted_path = ADAPTED_DIR / f"{self.name}.jsonl"
        if adapted_path.exists():
            return load_raw_jsonl(adapted_path)

        raw_path = RAW_DIR / f"{self.name}.jsonl"
        if raw_path.exists():
            ds = load_raw_jsonl(raw_path)
        else:
            ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
            save_jsonl(raw_path, ds)

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
        save_jsonl(adapted_path, final_records)
        return final_records

__all__ = ["TypedDecisionsSource"]
