import random
import re
from typing import List, Optional, Sequence
from data.case import DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

class GlaiveSource(IDataSource):
    """Real Glaive function-calling extractor: parses tool calls and yields tagged source records."""
    name = "glaive_tools"

    def __init__(self, seed: int = 42, test_ratio: float = 0.1, cache: Optional[IRawCachePolicy] = None):
        self.seed = seed
        self.test_ratio = test_ratio
        self.cache = cache or HFCache()

    def cases(self) -> Sequence[DecisionCase]:
        rng = random.Random(self.seed)
        rows = self.cache.load("glaiveai/glaive-function-calling-v2", split="train")

        cases: List[DecisionCase] = []
        for row in rows:
            system, chat = row.get("system", ""), row.get("chat", "")
            fn_match = re.search(r'<functioncall>\s*\{\s*"name":\s*"([^"]+)"', chat)
            user_match = re.search(r'USER:\s*(.*?)(?=\n\s*(?:ASSISTANT:|<\|endoftext\|>|$))', chat, re.DOTALL)
            if not (fn_match and user_match):
                continue
            fn_name, user_query = fn_match.group(1), user_match.group(1).strip()
            tools = dict(re.findall(r'\{\s*"name":\s*"([^"]+)",\s*"description":\s*"([^"]+)"', system))
            if fn_name not in tools or len(tools) < 2:
                continue

            cands_list = list(tools.keys())
            rng.shuffle(cands_list)
            cands_tuple = tuple(cands_list)
            probs = {c: (1.0 if c == fn_name else 0.0) for c in cands_tuple}
            is_test = rng.random() < self.test_ratio

            case = DecisionCase(
                state=user_query,
                candidates=cands_tuple,
                probabilities=probs,
                category="function_calling",
                source=self.name,
                split="test" if is_test else "train",
            )
            cases.append(case)
        return cases
