import random
import re
from typing import List, Sequence
from datasets import load_dataset
from inference.case import DecisionCase
from inference.questions import Choice
from data.sources import RAW_DIR, ADAPTED_DIR, IDataSource, SourceRecord, cached_jsonl, cached_records

class GlaiveSource(IDataSource):
    """Real Glaive function-calling extractor: parses tool calls and yields tagged source records."""
    name = "glaive_tools"

    def __init__(self, limit: int = 5000, seed: int = 42):
        self.limit = limit
        self.rng = random.Random(seed)

    def read(self) -> Sequence[SourceRecord]:
        def _fetch() -> List[dict]:
            try:
                ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
                return [item for _, item in zip(range(self.limit * 2), ds)]
            except Exception:
                return []

        def _build() -> List[SourceRecord]:
            rows = cached_jsonl(RAW_DIR / f"{self.name}.jsonl", _fetch)
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
                if len(raw_pairs) >= self.limit * 2:
                    break

            records: List[SourceRecord] = []
            all_names = list(tool_registry.keys())
            for query, fn, sample_tools in raw_pairs:
                candidates = dict(sample_tools)
                if len(candidates) < 5 and len(all_names) >= 5:
                    distractors = [n for n in all_names if n != fn and n not in candidates]
                    sample_k = min(len(distractors), 5 - len(candidates))
                    if sample_k > 0:
                        for d in self.rng.sample(distractors, sample_k):
                            candidates[d] = tool_registry[d]
                if fn in candidates:
                    sample = DecisionCase(state=query, questions=[Choice(name="tool", options=candidates, instruction="Select the tool to invoke:", target=fn)])
                    records.append(SourceRecord(sample=sample, tag="unsplit"))
                if len(records) >= self.limit:
                    break
            return records

        return cached_records(ADAPTED_DIR / f"{self.name}.jsonl", _build)

__all__ = ["GlaiveSource"]
