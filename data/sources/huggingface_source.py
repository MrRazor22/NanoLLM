from typing import List, Mapping, Optional, Sequence, Union
from datasets import load_dataset
from inference.case import DecisionCase
from inference.questions import Choice
from data.sources import RAW_DIR, ADAPTED_DIR, IDataSource, SourceRecord, cached_jsonl, cached_records

LabelKey = Union[str, int, bool]

class HuggingFaceSource(IDataSource):
    """Declarative HuggingFace dataset adapter: extracts cases with real class candidates, tags origin split, and caches adapted rows."""

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
        category: Optional[str] = None,
        candidate_options: Optional[Sequence[str]] = None,
        label_map: Optional[Mapping[LabelKey, str]] = None,
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
        self.category = category or dataset_name.split("/")[-1].replace("-", "_")
        self.candidate_options = list(candidate_options) if candidate_options is not None else None
        self.label_map = dict(label_map) if label_map is not None else None

    def read(self) -> Sequence[SourceRecord]:
        def _fetch() -> List[dict]:
            try:
                ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
                return [item for _, item in zip(range(self.limit), ds)]
            except Exception:
                return []

        def _build() -> List[SourceRecord]:
            rows = cached_jsonl(RAW_DIR / f"{self.name}_{self.split}.jsonl", _fetch)
            if not rows:
                return []

            if self.candidate_options:
                all_options = self.candidate_options
            else:
                raw_labels = set()
                for item in rows:
                    val = item.get(self.label_field)
                    if val is None:
                        val = item.get("label", item.get("queue"))
                    if val is not None:
                        l_str = self.label_map.get(val, val) if self.label_map else val
                        raw_labels.add(str(l_str).replace("_", " ").strip())
                all_options = sorted(list(raw_labels))

            records: List[SourceRecord] = []
            for item in rows:
                text = str(item.get(self.text_field) or item.get("prompt") or item.get("body") or "").strip()
                if not text:
                    continue
                raw_lbl = item.get(self.label_field)
                if raw_lbl is None:
                    raw_lbl = item.get("label", item.get("queue", "general"))
                lbl_mapped = self.label_map.get(raw_lbl, raw_lbl) if self.label_map else raw_lbl
                label_str = str(lbl_mapped).replace("_", " ").strip()

                options = all_options
                if label_str not in options:
                    options = sorted(list(set(options + [label_str])))

                sample = DecisionCase(
                    state=text[:2000],
                    questions=[Choice(name=self.question_name, options=options, instruction=self.instruction, target=label_str)],
                    category=self.category,
                )
                records.append(SourceRecord(sample=sample, tag=self.tag))
                if len(records) >= self.limit:
                    break
            return records

        return cached_records(ADAPTED_DIR / f"{self.name}_{self.split}.jsonl", _build)

__all__ = ["HuggingFaceSource"]
