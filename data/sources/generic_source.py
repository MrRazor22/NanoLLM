import json
from pathlib import Path
from typing import List, Optional, Sequence
from datasets import load_dataset
from inference.questions import Choice, DecisionSample
from data.sources import RAW_DIR, ADAPTED_DIR, IDataSource, SourceRecord, cached_jsonl

class GenericSource(IDataSource):
    """Declarative HuggingFace dataset adapter: extracts samples, tags origin split, and caches adapted rows."""

    def __init__(
        self,
        dataset_name: str,
        question_name: str,
        instruction: str,
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
        adapted_path = ADAPTED_DIR / f"{self.name}_{self.split}.jsonl"
        if adapted_path.exists():
            records: List[SourceRecord] = []
            for line in adapted_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                opts = d.get("options", ["true", "false"])
                target = d.get("target")
                sample = DecisionSample(
                    state=d["state"],
                    questions=[Choice(name=self.question_name, options=opts, instruction=self.instruction, target=target)],
                )
                records.append(SourceRecord(sample=sample, tag=d.get("tag", self.tag)))
            return records

        def _fetch():
            try:
                ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
                res = []
                for item in ds:
                    res.append(item)
                    if len(res) >= self.limit:
                        break
                return res
            except Exception:
                return []

        rows = cached_jsonl(RAW_DIR / f"{self.name}_{self.split}.jsonl", _fetch)
        records: List[SourceRecord] = []
        cache_rows: List[dict] = []

        for item in rows:
            text = str(item.get(self.text_field) or item.get("prompt") or item.get("body") or "")
            if not text.strip():
                continue
            lbl = item.get(self.label_field) or item.get("label") or item.get("queue") or "general"
            label_str = str(lbl).replace("_", " ")
            sample = DecisionSample(
                state=text[:2000],
                questions=[Choice(name=self.question_name, options=[label_str, "other"], instruction=self.instruction, target=label_str)],
            )
            records.append(SourceRecord(sample=sample, tag=self.tag))
            cache_rows.append({"state": sample.state, "options": [label_str, "other"], "target": label_str, "tag": self.tag})
            if len(records) >= self.limit:
                break

        if cache_rows:
            adapted_path.parent.mkdir(parents=True, exist_ok=True)
            adapted_path.write_text("\n".join(json.dumps(r) for r in cache_rows) + "\n", encoding="utf-8")

        return records

__all__ = ["GenericSource"]
