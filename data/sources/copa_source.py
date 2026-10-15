from typing import List, Sequence
from datasets import load_dataset
from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import RAW_DIR, read_raw_jsonl

class COPASource(IDataSource):
    """Choice of Plausible Alternatives (COPA) Source: causal directionality and root cause deduction."""
    name = "superglue_copa"

    def __init__(self, train_limit: int = 500, test_limit: int = 200):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("validation", "test")]:
            limit = self.test_limit if split_name in ("validation", "test") else self.train_limit
            target = RAW_DIR / f"copa_{split_name}.jsonl"
            rows = read_raw_jsonl(
                target,
                lambda s=split_name: list(load_dataset("aps/super_glue", "copa", split=s)),
            )
            if limit is not None and len(rows) > limit:
                rows = rows[:limit]

            for r in rows:
                premise = str(r.get("premise", "")).strip()
                choice1 = str(r.get("choice1", "")).strip()
                choice2 = str(r.get("choice2", "")).strip()
                q_type = str(r.get("question", "cause")).strip().lower()
                label_raw = r.get("label")
                if not premise or not choice1 or not choice2 or label_raw is None:
                    continue
                try:
                    target_idx = int(label_raw)
                except ValueError:
                    continue
                options = [choice1, choice2]
                if target_idx < 0 or target_idx >= len(options):
                    continue
                instruction = "What was the cause of the premise?" if q_type == "cause" else "What was the effect of the premise?"
                records.append(SourceRecord(
                    sample=DecisionCase(
                        state=premise,
                        questions=[Choice(name="causal_link", options=options, instruction=instruction, target=options[target_idx])],
                        category="causal_reasoning",
                    ),
                    tag=tag,
                ))
        return records

__all__ = ["COPASource"]
