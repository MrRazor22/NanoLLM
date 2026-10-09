import json
from pathlib import Path
from typing import List, Optional, Sequence
from datasets import load_dataset
from piqo import Choice, DecisionSample
from harness.sources import RAW_DIR, ISource, cached_jsonl

class DecisionsSource(ISource):
    """Real Typed-Decisions dataset extractor: loads multi-question typed decision benchmarks."""
    name = "typed_decisions"

    def __init__(self, repeat: int = 5):
        self.repeat = repeat

    def load(self, limit: Optional[int] = None) -> Sequence[DecisionSample]:
        rows = cached_jsonl(
            RAW_DIR / f"{self.name}.jsonl",
            lambda: list(load_dataset("LocalLLaMA/typed-decisions", "all", split="train"))
        )

        samples: List[DecisionSample] = []
        for row in rows:
            q_defs = row.get("questions", {})
            g_defs = row.get("gold", {})
            q_dict = json.loads(q_defs) if isinstance(q_defs, str) else q_defs
            g_dict = json.loads(g_defs) if isinstance(g_defs, str) else g_defs
            qs = []
            for qid, q_data in q_dict.items():
                if q_data.get("type") == "choice":
                    qs.append(
                        Choice(
                            name=qid,
                            options=q_data.get("criteria", {}),
                            instruction=q_data.get("instructions", "Select the best option:"),
                            target=g_dict.get(qid, {}).get("label"),
                        )
                    )
            if qs:
                samples.append(DecisionSample(state=row.get("state", ""), questions=qs))
            if limit and len(samples) >= limit:
                break
        return samples * self.repeat

__all__ = ["DecisionsSource"]
