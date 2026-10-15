from typing import List, Sequence
from datasets import load_dataset

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import RAW_DIR, read_raw_jsonl

HELLASWAG_INSTRUCTION = "Which ending logically completes the scenario?"

def _load_hellaswag_split(split: str, limit: int) -> List[dict]:
    target = RAW_DIR / f"hellaswag_{split}.jsonl"

    def _fetch() -> List[dict]:
        rows = list(load_dataset("Rowan/hellaswag", split=split))
        if limit is not None:
            rows = rows[:limit]
        return rows

    rows = read_raw_jsonl(target, _fetch)
    if limit is not None and len(rows) > limit:
        rows = rows[:limit]
    return rows

class HellaSwagSource(IDataSource):
    """Adversarially Filtered Grounding Source: extracts HellaSwag continuation decisions."""
    name = "rowan_hellaswag"

    def __init__(self, train_limit: int = 2000, test_limit: int = 300):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("validation", "test")]:
            limit = self.test_limit if split_name == "validation" else self.train_limit
            rows = _load_hellaswag_split(split_name, limit)
            for r in rows:
                state = str(r.get("ctx", "")).strip()
                endings = r.get("endings", [])
                label_raw = r.get("label")

                if not state or not endings or label_raw is None:
                    continue

                try:
                    target_idx = int(label_raw)
                except ValueError:
                    continue

                if target_idx < 0 or target_idx >= len(endings):
                    continue

                cleaned_endings = [e.lstrip(", ").strip() for e in endings]
                target_str = cleaned_endings[target_idx]

                case = DecisionCase(
                    state=state,
                    questions=[
                        Choice(
                            name="ending",
                            options=cleaned_endings,
                            instruction=HELLASWAG_INSTRUCTION,
                            target=target_str,
                        )
                    ],
                    category="adversarial_grounding",
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["HellaSwagSource", "HELLASWAG_INSTRUCTION"]
