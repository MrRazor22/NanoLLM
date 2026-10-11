from typing import List, Sequence
from datasets import load_dataset
from inference.case import DecisionCase
from inference.questions import Choice
from data.sources import RAW_DIR, ADAPTED_DIR, IDataSource, SourceRecord, cached_jsonl, cached_records

class GenericSource(IDataSource):
    """Declarative HuggingFace dataset adapter: extracts cases, tags origin split, and caches adapted rows."""

    def __init__(
        self,
        dataset_name: str = "mteb/banking77",
        question_name: str = "intent",
        instruction: str = "What is the primary customer request?",
        limit: int = 4000,
        text_field: str = "text",
        label_field: str = "label_text",
        split: str = "train",
        tag: str = "unsplit",
    ):
        self.name = dataset_name.replace("/", "_")
        self.dataset_name = dataset_name
        self.question_name = question_name
        self.instruction = instruction
        self.limit = limit
        self.text_field = text_field
        self.label_field = label_field
        self.split = split
        self.tag = tag

    def read(self) -> Sequence[SourceRecord]:
        def _fetch() -> List[dict]:
            try:
                ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
                return [item for _, item in zip(range(self.limit), ds)]
            except Exception:
                return []

        def _build() -> List[SourceRecord]:
            rows = cached_jsonl(RAW_DIR / f"{self.name}_{self.split}.jsonl", _fetch)
            records: List[SourceRecord] = []
            for item in rows:
                text = str(item.get(self.text_field) or item.get("prompt") or item.get("body") or "").strip()
                if not text:
                    continue
                lbl = item.get(self.label_field) or item.get("label") or item.get("queue") or "general"
                label_str = str(lbl).replace("_", " ")
                cat = self.name.split("/")[-1].replace("-", "_")
                sample = DecisionCase(
                    state=text[:2000],
                    questions=[Choice(name=self.question_name, options=[label_str, "other"], instruction=self.instruction, target=label_str)],
                    category=cat,
                )
                records.append(SourceRecord(sample=sample, tag=self.tag))
                if len(records) >= self.limit:
                    break
            return records

        return cached_records(ADAPTED_DIR / f"{self.name}_{self.split}.jsonl", _build)

__all__ = ["GenericSource"]
