from typing import List, Sequence

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

SCIQ_INSTRUCTION = "Which option correctly answers the question?"

class SciQSource(IDataSource):
    """Scientific Mechanism & Distractor Elimination Source: scientific reasoning with negative distractors."""
    name = "allenai_sciq"

    def __init__(self, train_limit: int = 1500, test_limit: int = 200):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("validation", "test")]:
            limit = self.test_limit if split_name in ("validation", "test") else self.train_limit
            rows = load_raw_jsonl(f"sciq_{split_name}", "allenai/sciq", split=split_name, limit=limit)
            for r in rows:
                question = str(r.get("question", "")).strip()
                correct = str(r.get("correct_answer", "")).strip()
                d1 = str(r.get("distractor1", "")).strip()
                d2 = str(r.get("distractor2", "")).strip()
                d3 = str(r.get("distractor3", "")).strip()

                if not question or not correct or not d1 or not d2 or not d3:
                    continue

                options = [correct, d1, d2, d3]
                if len(set(options)) < 4:
                    continue

                case = DecisionCase(
                    state=question,
                    questions=[
                        Choice(
                            name="answer",
                            options=options,
                            instruction=SCIQ_INSTRUCTION,
                            target=correct,
                        )
                    ],
                    category="scientific_reasoning",
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["SciQSource", "SCIQ_INSTRUCTION"]
