from typing import List, Sequence

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

class ReClorSource(IDataSource):
    """ReClor Logical Reasoning Source: LSAT multi-constraint argument analysis and fallacy elimination."""
    name = "reclor"

    def __init__(self, train_limit: int = 2000, test_limit: int = 400):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("validation", "test")]:
            limit = self.test_limit if tag == "test" else self.train_limit
            rows = load_raw_jsonl(f"reclor_{split_name}", "metaeval/reclor", split=split_name, limit=limit)
            for r in rows:
                context = str(r.get("context", "")).strip()
                question = str(r.get("question", "")).strip()
                answers = r.get("answers", [])
                label = r.get("label")

                if not context or not answers or len(answers) < 2 or label is None:
                    continue

                try:
                    idx = int(label)
                except (ValueError, TypeError):
                    continue

                if idx < 0 or idx >= len(answers):
                    continue

                state = f"Context: {context}\nQuestion: {question}" if question else context
                case = DecisionCase(
                    state=state,
                    questions=[
                        Choice(
                            name="logical_deduction",
                            options=answers,
                            instruction=question or "Which conclusion logically follows?",
                            target=answers[idx],
                        )
                    ],
                    category="logical_reasoning",
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["ReClorSource"]
