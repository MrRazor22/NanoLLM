import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from harness.dataset.training_dataset import ADAPTED_DIR, RAW_DIR, IDataSource

class GlaiveToolSource(IDataSource):
    name = "glaive_tools"
    def __init__(self, limit: int = 5000, rng: Optional[random.Random] = None):
        self.limit = limit
        self.rng = rng or random.Random(42)

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
            streaming_ds = rows
        else:
            streaming_ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            cached_rows = []
            for row in streaming_ds:
                cached_rows.append(row)
                if len(cached_rows) >= self.limit:
                    break
            with open(raw_path, "w", encoding="utf-8") as f:
                for r in cached_rows:
                    f.write(json.dumps(r, default=str) + "\n")
            streaming_ds = cached_rows
        raw, tool_registry = [], {}
        for row in streaming_ds:
            if len(raw) >= self.limit:
                break
            txt = row.get("chat", "")
            if not txt.startswith("SYSTEM:"):
                continue
            parts = txt.split("USER:")
            if len(parts) < 2:
                continue
            sys_part = parts[0].replace("SYSTEM:", "").strip()
            rest = parts[1]
            u_parts = rest.split("ASSISTANT:")
            user_part = u_parts[0].strip()
            asst_part = u_parts[1].strip() if len(u_parts) > 1 else ""
            call_match = re.search(r"<functioncall>\s*(\{.*?\})", asst_part)
            if not call_match:
                continue
            for fn in re.findall(r"\{\s*\"name\"\s*:\s*\"([^\"]+)\"", sys_part):
                if fn not in tool_registry:
                    tool_registry[fn] = f"Tool function: {fn}"
            raw.append({"user": user_part, "gold_call": call_match.group(1), "tools": list(tool_registry.keys())})

        records = []
        for r in raw:
            match = re.search(r"\"name\"\s*:\s*\"([^\"]+)\"", r["gold_call"])
            if not match:
                continue
            gold_tool = match.group(1)
            distractors = [t for t in tool_registry.keys() if t != gold_tool]
            self.rng.shuffle(distractors)
            selected_tools = [gold_tool] + distractors[:4]
            self.rng.shuffle(selected_tools)
            criteria = {t: tool_registry.get(t, f"Tool: {t}") for t in selected_tools}
            records.append({
                "category": "glaive_tools",
                "state": f"User Request: {r['user']}",
                "questions": {"tool": {"type": "choice", "instructions": "Select the appropriate tool for user request.", "criteria": criteria}},
                "gold": {"tool": {"type": "choice", "label": gold_tool}},
            })

        ADAPTED_DIR.mkdir(parents=True, exist_ok=True)
        with open(adapted_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return records

__all__ = ["GlaiveToolSource"]
