import json
from pathlib import Path
import random
import re
from typing import List, Sequence
from datasets import load_dataset
from data.case import DecisionCase, IDataSource, SourceRecord
from inference.questions import Choice
from data.sources.huggingface_source import RAW_DIR, read_raw_jsonl

class GlaiveSource(IDataSource):
    """Real Glaive function-calling extractor: parses tool calls and yields tagged source records."""
    name = "glaive_tools"

    def __init__(self, seed: int = 42, test_ratio: float = 0.1):
        self.seed = seed
        self.test_ratio = test_ratio

    @property
    def signature(self) -> str:
        import hashlib
        return hashlib.sha256(f"glaive_{self.seed}_{self.test_ratio}".encode()).hexdigest()[:10]

    def read(self) -> Sequence[SourceRecord]:
        rng = random.Random(self.seed)
        def _fetch() -> List[dict]:
            return list(load_dataset("glaiveai/glaive-function-calling-v2", split="train"))

        rows = read_raw_jsonl(RAW_DIR / f"{self.name}.jsonl", _fetch)

        raw_pairs, tool_registry = [], {}
        for row in rows:
            system, chat = row.get("system", ""), row.get("chat", "")
            fn_match = re.search(r'<functioncall>\s*\{\s*"name":\s*"([^"]+)"', chat)
            user_match = re.search(r'USER:\s*(.*?)(?=\n\s*(?:ASSISTANT:|<\|endoftext\|>|$))', chat, re.DOTALL)
            if not (fn_match and user_match):
                continue
            fn_name, user_query = fn_match.group(1), user_match.group(1).strip()
            tools = dict(re.findall(r'\{\s*"name":\s*"([^"]+)",\s*"description":\s*"([^"]+)"', system))
            if fn_name not in tools:
                continue
            tool_registry.update(tools)
            raw_pairs.append((user_query, fn_name, tools))

        cases: List[DecisionCase] = []
        all_names = list(tool_registry.keys())
        for query, fn, sample_tools in raw_pairs:
            candidates = dict(sample_tools)
            if len(candidates) < 5 and len(all_names) >= 5:
                distractors = [n for n in all_names if n != fn and n not in candidates]
                sample_k = min(len(distractors), 5 - len(candidates))
                if sample_k > 0:
                    for d in rng.sample(distractors, sample_k):
                        candidates[d] = tool_registry[d]
            if fn in candidates:
                cases.append(DecisionCase(
                    state=query,
                    questions=[Choice(name="tool", options=candidates, instruction="Select the tool to invoke:", target=fn)],
                    category="function_calling",
                ))

        rng.shuffle(cases)
        test_count = max(1, int(len(cases) * self.test_ratio)) if cases else 0
        records: List[SourceRecord] = []
        for idx, c in enumerate(cases):
            tag = "test" if idx < test_count else "train"
            records.append(SourceRecord(sample=c, tag=tag))
        return records

__all__ = ["GlaiveSource"]
