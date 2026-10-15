from typing import List, Sequence

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

HELLASWAG_INSTRUCTION = "Which ending logically completes the scenario?"

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
            rows = load_raw_jsonl(f"hellaswag_{split_name}", "Rowan/hellaswag", split=split_name, limit=limit)
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
