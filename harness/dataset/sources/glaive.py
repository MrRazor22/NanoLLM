import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from harness.dataset.training_dataset import RAW_DIR, IDataSource
from harness.dataset.transforms import load_raw_jsonl, save_jsonl

class GlaiveToolSource(IDataSource):
    name = "glaive_tools"

    def __init__(self, limit: int = 5000, rng: Optional[random.Random] = None):
        self.limit = limit
        self.rng = rng or random.Random(42)

    def extract(self) -> List[Dict[str, Any]]:
        raw_path = RAW_DIR / f"{self.name}.jsonl"
        if raw_path.exists():
            streaming_ds = load_raw_jsonl(raw_path)
        else:
            ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
            streaming_ds = []
            for row in ds:
                streaming_ds.append(row)
                if len(streaming_ds) >= self.limit * 2:
                    break
            save_jsonl(raw_path, streaming_ds)

        raw, tool_registry = [], {}
        for row in streaming_ds:
            system = row.get("system", "")
            chat = row.get("chat", "")
            fn_match = re.search(r'<functioncall>\s*\{\s*"name":\s*"([^"]+)"', chat)
            user_match = re.search(r'USER:\s*(.*?)(?=\n\s*(?:ASSISTANT:|<\|endoftext\|>|$))', chat, re.DOTALL)
            if not (fn_match and user_match):
                continue
            fn_name = fn_match.group(1)
            user_query = user_match.group(1).strip()
            tools = dict(re.findall(r'\{\s*"name":\s*"([^"]+)",\s*"description":\s*"([^"]+)"', system))
            if fn_name not in tools:
                continue
            tool_registry.update(tools)
            raw.append((user_query, fn_name, tools))
            if len(raw) >= self.limit * 2:
                break

        records = []
        all_names = list(tool_registry.keys())
        for query, fn, sample_tools in raw:
            candidates = dict(sample_tools)
            if len(candidates) < 5 and len(all_names) >= 5:
                distractors = [n for n in all_names if n != fn and n not in candidates]
                sample_k = min(len(distractors), 5 - len(candidates))
                if sample_k > 0:
                    for d in self.rng.sample(distractors, sample_k):
                        candidates[d] = tool_registry[d]
            opt_keys = list(candidates.keys())
            self.rng.shuffle(opt_keys)
            prompt_variations = [
                "Which function or tool should be invoked to handle this request?",
                "Which production tool must the agent select to resolve this situation?",
                "Select the appropriate tool or function for this task:",
            ]
            instr = prompt_variations[len(records) % len(prompt_variations)]
            records.append({
                "category": "agent_tool_routing",
                "state": query,
                "questions": {
                    "tool": {
                        "type": "choice",
                        "instructions": instr,
                        "criteria": {k: candidates[k] for k in opt_keys},
                    }
                },
                "gold": {"tool": {"type": "choice", "label": fn}},
            })
            if len(records) >= self.limit:
                break

        return records

__all__ = ["GlaiveToolSource"]
