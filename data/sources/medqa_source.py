from typing import List, Sequence
from datasets import load_dataset

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import RAW_DIR, read_raw_jsonl

MEDQA_INSTRUCTION = "What is the most likely diagnosis, mechanism, or correct next step?"

def _load_medqa_split(split: str, limit: int) -> List[dict]:
    target = RAW_DIR / f"medqa_{split}.jsonl"

    def _fetch() -> List[dict]:
        rows = list(load_dataset("GBaker/MedQA-USMLE-4-options", split=split))
        if limit is not None:
            rows = rows[:limit]
        return rows

    rows = read_raw_jsonl(target, _fetch)
    if limit is not None and len(rows) > limit:
        rows = rows[:limit]
    return rows

class MedQASource(IDataSource):
    """Differential Elimination Source: extracts multi-variable clinical elimination decisions."""
    name = "gbaker_medqa"

    def __init__(self, train_limit: int = 2000, test_limit: int = 300):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name in ("train", "test"):
            limit = self.test_limit if split_name == "test" else self.train_limit
            rows = _load_medqa_split(split_name, limit)
            tag = "test" if split_name == "test" else "train"
            for r in rows:
                state = str(r.get("question", "")).strip()
                opts_dict = r.get("options", {})
                answer_str = str(r.get("answer", "")).strip()

                if not state or not opts_dict or not answer_str:
                    continue

                options_list = list(opts_dict.values())
                if len(options_list) < 2 or answer_str not in options_list:
                    continue

                case = DecisionCase(
                    state=state,
                    questions=[
                        Choice(
                            name="diagnosis",
                            options=options_list,
                            instruction=MEDQA_INSTRUCTION,
                            target=answer_str,
                        )
                    ],
                    category="differential_elimination",
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["MedQASource", "MEDQA_INSTRUCTION"]
