import json
from pathlib import Path
import random
import re
from typing import Dict, List, Sequence
from datasets import load_dataset
from inference.case import DecisionCase
from inference.questions import Choice
from data.sources import RAW_DIR, ADAPTED_DIR, IDataSource, SourceRecord, cached_jsonl

class GlaiveSource(IDataSource):
    """Real Glaive function-calling extractor: parses tool calls and yields tagged source records."""
    name = "glaive_tools"

    def __init__(self, limit: int = 5000, seed: int = 42):
        self.limit = limit
        self.rng = random.Random(seed)

    def read(self) -> Sequence[SourceRecord]:
        adapted_path = ADAPTED_DIR / f"{self.name}.jsonl"
        if adapted_path.exists():
            records: List[SourceRecord] = []
            for line in adapted_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                sample = DecisionCase(
                    state=d["state"],
                    questions=[Choice(name="tool", options=d["options"], instruction="Select the tool to invoke:", target=d["target"])],
                )
                records.append(SourceRecord(sample=sample, tag="unsplit"))
            return records

        def _fetch():
            try:
                ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
                res = []
                for item in ds:
                    res.append(item)
                    if len(res) >= self.limit * 2:
                        break
                return res
            except Exception:
                return []

        rows = cached_jsonl(RAW_DIR / f"{self.name}.jsonl", _fetch)
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
            if len(raw_pairs) >= self.limit * 2:
                break

        records: List[SourceRecord] = []
        cache_rows: List[dict] = []
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
            if fn in opt_keys:
                sample = DecisionCase(
                    state=query,
                    questions=[Choice(name="tool", options=candidates, instruction="Select the tool to invoke:", target=fn)],
                )
                records.append(SourceRecord(sample=sample, tag="unsplit"))
                cache_rows.append({"state": query, "options": candidates, "target": fn})
            if len(records) >= self.limit:
                break

        if cache_rows:
            adapted_path.parent.mkdir(parents=True, exist_ok=True)
            adapted_path.write_text("\n".join(json.dumps(r) for r in cache_rows) + "\n", encoding="utf-8")

        return records

__all__ = ["GlaiveSource"]
