import random
import re
from typing import List, Sequence
from data.case import ChoiceAnswer, ChoiceOption, DecisionCase
from data.sources.source import IDataSource
from data.sources.huggingface_source import load_raw_jsonl

class GlaiveSource(IDataSource):
    """Real Glaive function-calling extractor: parses tool calls and yields tagged source records."""
    name = "glaive_tools"

    def __init__(self, seed: int = 42, test_ratio: float = 0.1):
        self.seed = seed
        self.test_ratio = test_ratio

    def cases(self) -> Sequence[DecisionCase]:
        rng = random.Random(self.seed)
        rows = load_raw_jsonl(self.name, "glaiveai/glaive-function-calling-v2", split="train")

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

        # Partition tools into strictly disjoint train and test subsets (Zero-Shot Tool Holdout)
        all_tools = sorted(list(tool_registry.keys()))
        rng.shuffle(all_tools)
        test_tool_count = max(1, int(len(all_tools) * self.test_ratio)) if all_tools else 0
        test_tool_set = set(all_tools[:test_tool_count])
        train_tool_set = set(all_tools[test_tool_count:])

        cases: List[DecisionCase] = []
        for query, fn, sample_tools in raw_pairs:
            is_test = fn in test_tool_set
            pool = test_tool_set if is_test else train_tool_set
            candidates = {k: v for k, v in sample_tools.items() if k in pool}
            candidates[fn] = tool_registry[fn]
            if len(candidates) < 5 and len(pool) >= 5:
                distractor_names = [n for n in pool if n != fn and n not in candidates]
                sample_k = min(len(distractor_names), 5 - len(candidates))
                if sample_k > 0:
                    for d in rng.sample(distractor_names, sample_k):
                        candidates[d] = tool_registry[d]

            case = DecisionCase(
                state=query,
                option=ChoiceOption(
                    options=candidates,
                    answer=ChoiceAnswer(choice=fn, confidence=1.0, probabilities={fn: 1.0}),
                ),
                category="function_calling",
                split="test" if is_test else "train",
                source=self.name,
            )
            cases.append(case)
        return cases
