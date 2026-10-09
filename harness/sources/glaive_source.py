import json
from pathlib import Path
import random
import re
from typing import Dict, List, Optional, Sequence
from datasets import load_dataset
from nanollm import Choice, DecisionSample
from harness.sources.sources import RAW_DIR, ISource

class GlaiveSource(ISource):
    """Real Glaive function-calling extractor: parses tool calls and injects negative distractors."""
    name = "glaive_tools"

    def __init__(self, limit: int = 5000, seed: int = 42):
        self.limit = limit
        self.rng = random.Random(seed)

    def load(self, limit: Optional[int] = None) -> Sequence[DecisionSample]:
        max_items = limit or self.limit
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_fp = RAW_DIR / f"{self.name}.jsonl"

        rows = []
        if raw_fp.exists():
            with open(raw_fp, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            try:
                ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
                for item in ds:
                    rows.append(item)
                    if len(rows) >= max_items * 2:
                        break
                with open(raw_fp, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
            except Exception:
                return []

        raw_pairs, tool_registry = [], {}
        for row in rows:
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
            raw_pairs.append((user_query, fn_name, tools))
            if len(raw_pairs) >= max_items * 2:
                break

        samples: List[DecisionSample] = []
        all_names = list(tool_registry.keys())
        for query, fn, sample_tools in raw_pairs:
            candidates = dict(sample_tools)
            if len(candidates) < 5 and len(all_names) >= 5:
                distractors = [n for n in all_names if n != fn and n not in candidates]
                sample_k = min(len(distractors), 5 - len(candidates))
                if sample_k > 0:
                    for d in self.rng.sample(distractors, sample_k):
                        candidates[d] = tool_registry[d]
            opt_keys = list(candidates.keys())
            self.rng.shuffle(opt_keys)
            samples.append(
                DecisionSample(
                    state=query,
                    questions=[
                        Choice(
                            name="tool",
                            options={k: candidates[k] for k in opt_keys},
                            instruction="Select the appropriate tool to execute:",
                            target=fn,
                        )
                    ],
                )
            )
            if len(samples) >= max_items:
                break
        return samples

__all__ = ["GlaiveSource"]
