import re
from typing import List, Optional, Sequence
from data.case import CaseMeta, DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

class GlaiveSource(IDataSource):
    def __init__(self, train_limit: Optional[int] = None, cache: Optional[IRawCachePolicy] = None):
        self.train_limit = train_limit
        self.cache = cache or HFCache()

    def cases(self) -> Sequence[DecisionCase]:
        rows = self.cache.load("glaiveai/glaive-function-calling-v2", split="train", limit=self.train_limit)

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

            cands_tuple = tuple(tools.keys())
            probs = {c: (1.0 if c == fn_name else 0.0) for c in cands_tuple}

            case = DecisionCase(
                state=user_query,
                candidates=cands_tuple,
                probabilities=probs,
                meta=CaseMeta(
                                candidate_type='choice',
                    category="function_calling",
                    source="glaive_tools",
                    split="train",
                ),
            )
            cases.append(case)
        return cases
