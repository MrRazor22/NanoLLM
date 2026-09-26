import random
import re
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from dataset.sources.generic import ISourceAdapter

class GlaiveToolSource(ISourceAdapter):
    def __init__(self, limit: int = 5000, rng: Optional[random.Random] = None):
        self.limit = limit
        self.rng = rng or random.Random(42)

    def extract(self) -> List[Dict[str, Any]]:
        streaming_ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
        raw, tool_registry = [], {}
        for row in streaming_ds:
            system, chat = row["system"], row["chat"]
            fn_match = re.search(r'<functioncall>\s*\{\s*"name":\s*"([^"]+)"', chat)
            user_match = re.search(r'USER:\s*(.*?)(?=\n\s*(?:ASSISTANT:|<\|endoftext\|>|$))', chat, re.DOTALL)
            if not (fn_match and user_match): continue
            fn_name, user_query = fn_match.group(1), user_match.group(1).strip()
            tools = dict(re.findall(r'\{\s*"name":\s*"([^"]+)",\s*"description":\s*"([^"]+)"', system))
            if fn_name not in tools: continue
            tool_registry.update(tools)
            raw.append((user_query, fn_name, tools))
            if len(raw) >= self.limit * 2: break

        records = []
        all_names = list(tool_registry.keys())
        for query, fn, sample_tools in raw:
            candidates = dict(sample_tools)
            if len(candidates) < 5 and len(all_names) >= 5:
                distractors = [n for n in all_names if n != fn and n not in candidates]
                for d in self.rng.sample(distractors, min(len(distractors), 5 - len(candidates))):
                    candidates[d] = tool_registry[d]
            opt_keys = list(candidates.keys())
            self.rng.shuffle(opt_keys)
            records.append({
                "state": query,
                "questions": [["tool", "choice", opt_keys.index(fn), {k: candidates[k] for k in opt_keys}, "Which function or tool should be invoked to handle this request?"]]
            })
            if len(records) >= self.limit: break
        return records
